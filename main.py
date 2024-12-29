import asyncio
import base64
import json
import os
from io import BytesIO
import time
from typing import Dict,  List, Tuple

from openai.types import Batch
from typing import Any
from system_prompt import system_prompt
from pdf2image import convert_from_path
from prisma import Prisma
from prisma.types import (
    DocumentUpsertInput,
)
from build_batch_line import build_batch_line
from process_batch_output import process_batch_output
from openai_client import client


JsonDict = Dict[str, Any]
InputDataType = Dict[str, Dict[str, Any]]
"""
Example shape:
{
   "doc_id_1": {
       "messages": ["...","..."],
       "pdf_filepath": "/path/to/cv.pdf"
   },
   "doc_id_2": {
       "messages": ["..."],
       "pdf_filepath": "/path/to/another.pdf"
   }
}
"""


async def connect_db() -> Prisma:
    """Connect to the Prisma-managed database."""
    db = Prisma()
    await db.connect()
    return db


async def disconnect_db(db: Prisma) -> None:
    """Disconnect from the database."""
    await db.disconnect()


def load_input_data(file_path: str) -> InputDataType:
    """
    Load the JSON from `file_path` and return the dict structure.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


async def process_document(
    db: Prisma,
    doc_id: str,
    doc_content: Dict[str, Any]
) -> str:
    """
    Convert PDF to PNG (if needed), store as DocumentBlob in DB,
    upsert a Document record, and return the base64 string for
    embedding in the prompt.
    """
    pdf_filepath = doc_content.get("pdf_filepath", "")
    png_bytes_b64 = ""

    # If the file is a PDF, convert first page -> PNG, store bytes in DB
    if pdf_filepath.lower().endswith(".pdf"):
        images = convert_from_path(pdf_filepath, dpi=200)
        if images:
            # Convert first page to PNG in memory
            buffer = BytesIO()
            images[0].save(buffer, format="PNG")
            buffer.seek(0)
            png_bytes = buffer.read()

            # Base64-encode for the model prompt
            base64_bytes = base64.b64encode(png_bytes)
            png_bytes_b64 = base64_bytes.decode("utf-8")

            # Upsert Document
            upsert_data: DocumentUpsertInput = {
                "create": {
                    "id": doc_id,
                    "mime_type": "application/pdf",
                    "documentBlob": png_bytes_b64,
                },
                "update": {}
            }
            upserted_data = await db.document.upsert(
                where={"id": doc_id},
                data=upsert_data
            )
            print(f"Upserted a Document record with ID: {upserted_data.id}")

    return png_bytes_b64


def write_batch_jsonl_file(batch_lines: List[str], filename: str) -> None:
    """
    Write the given list of JSON lines to `filename` with newlines in between.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(batch_lines))
        f.write("\n")  # Ensure there's a trailing newline


def create_and_submit_batch(batch_file: str) -> str:
    """
    Upload the JSONL batch_file to OpenAI, create a Batch, and return batch_id.
    """
    print(f"Uploading batch input file '{batch_file}' to OpenAI...")
    batch_input_upload = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )

    print("Creating batch job...")
    created_batch = client.batches.create(
        input_file_id=batch_input_upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Nightly CV transcription job"
        }
    )

    batch_id = created_batch.id
    print('Created a batch with id: ', created_batch.id)
    return batch_id


def poll_batches_until_done(
    batch_ids: List[str], 
    sleep_seconds: int = 10
) -> List[Tuple[str, Batch]]:
    """
    Poll multiple batch IDs. Keep checking until each is either completed,
    failed, canceled, or expired. Returns a list of (batch_id, final_batch_object).
    """

    # We store final results in a dict {batch_id: BatchObject}:
    results = {}
    remaining = set(batch_ids)

    while remaining:
        # We'll copy to avoid changing the set while iterating
        for b_id in list(remaining):
            batch = client.batches.retrieve(b_id)

            status = batch.status
            print(f"Batch {b_id} status: {status}")
            if status in ["completed", "failed", "canceled", "expired"]:
                # store final result
                results[b_id] = batch
                remaining.remove(b_id)

        if remaining:
            print(f"Still waiting on {len(remaining)} batch(es): {list(remaining)}")
            print(f"Sleeping for {sleep_seconds} seconds before next poll...")
            time.sleep(sleep_seconds)

    # Return a list of (batch_id, batch) for the final results
    return list(results.items())


async def process_completed_batches(
    db: Prisma,
    final_batches: List[Tuple[str, Batch]],
    api_call_result_dir: str
) -> None:
    """
    For each completed batch, download & process results using process_batch_output.
    Skips processing for batches that didn't end in 'completed'.
    """
    os.makedirs(api_call_result_dir, exist_ok=True)

    for (batch_id, batch_obj) in final_batches:
        if batch_obj.status == "completed":
            output_filename = os.path.join(api_call_result_dir, f"out-{batch_id}.jsonl")
            await process_batch_output(db, batch_obj, output_filename)
        else:
            print(f"Batch {batch_id} ended with status '{batch_obj.status}'. Skipping processing.")


def chunk_batch_lines(
    all_lines: List[str],
    max_batch_file_size_bytes: int = 200 * 1024 * 1024
) -> List[List[str]]:
    """
    Splits a list of JSON lines into smaller chunks so that each chunk's total
    byte-size does not exceed `max_batch_file_size_bytes`.

    Note: Each line is assumed to be valid JSON. We'll measure total chunk size by
    summing the UTF-8 byte length of each line. This doesn't account for any overhead
    from newlines, but is generally close enough. If you're borderline near 200MB,
    consider lowering the threshold.
    """
    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_size = 0

    for line in all_lines:
        encoded_line_size = len(line.encode("utf-8"))
        # Add 1 for the newline overhead
        overhead = 1

        if current_chunk and (current_size + encoded_line_size + overhead) > max_batch_file_size_bytes:
            # If adding this line would exceed the max, start a new chunk
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(line)
        current_size += (encoded_line_size + overhead)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ----------------------------------------------------------------------------
# 7. ORCHESTRATION (MAIN)
# ----------------------------------------------------------------------------
async def main():
    """
    1) Connect to DB.
    2) Load input data (doc_id -> { messages, pdf_filepath }).
    3) For each doc, create PNG (if PDF), store in DB as base64, build
       prompt line -> collect all in 'all_batch_lines'.
    4) Split those lines into multiple .jsonl chunk files if they exceed ~200MB.
    5) Create a separate OpenAI batch for each chunk file.
    6) Poll all batch jobs until done.
    7) For each completed batch, retrieve & process output (store in DB).
    8) Disconnect from DB.
    """
    # 7.1 Connect DB & load environment variables
    db = await connect_db()

    # We assume this JSON file contains the "doc_id -> {...}" mapping
    input_json_path = "json_files/pdf_children_texts_fBXQI.json"

    # 7.2 Read input data
    data: InputDataType = load_input_data(input_json_path)

    # 7.3 For each document, process PDF->PNG, store in DB, build user content
    all_batch_lines = []
    model_name = "gpt-4o-mini"

    for doc_id, doc_content in data.items():
        # 7.3.1 Convert PDF -> PNG, store DocumentBlob/Document
        png_b64 = await process_document(db, doc_id, doc_content)

        # 7.3.2 Merge user messages into single string
        messages_list = doc_content.get("messages", [])
        review_conversation = "\n".join(str(m) for m in messages_list)

        # 7.3.3 Build one .jsonl line for the OpenAI batch
        line = build_batch_line(
            doc_id=doc_id,
            review_conversation=review_conversation,
            png_bytes_b64=png_b64,
            system_prompt=system_prompt,
            model=model_name
        )
        all_batch_lines.append(line)

    # 7.3.4 Split the lines into multiple chunked lists if the size is too large
    chunked_batches = chunk_batch_lines(all_batch_lines, max_batch_file_size_bytes=200*1024*1024)
    batch_dir = 'batches'
    os.makedirs(batch_dir, exist_ok=True)

    # 7.4 Write each chunk to its own .jsonl file, then create & track the batch
    batch_ids: List[str] = []

    for i, chunk_lines in enumerate(chunked_batches, start=1):
        batch_input_file = os.path.join(batch_dir, f"batchinput_{i}.jsonl")
        write_batch_jsonl_file(chunk_lines, batch_input_file)
        batch_id = create_and_submit_batch(batch_input_file)
        batch_ids.append(batch_id)

    # 7.5 Poll all batches until done
    final_batches = poll_batches_until_done(batch_ids, sleep_seconds=10)

    # 7.6 Process results from completed batches
    api_call_result_dir = 'api_call_results'
    await process_completed_batches(db, final_batches, api_call_result_dir)

    # 7.7 Disconnect
    await disconnect_db(db)


if __name__ == "__main__":
    asyncio.run(main())
