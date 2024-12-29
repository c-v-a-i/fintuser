import asyncio
import base64
import json
import os
from io import BytesIO
import time
from typing import Dict

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
                    # "messages": [] // messages should only appear here after the processing. Though it might be stupid, but whatever
                },
                "update": {}
            }
            upserted_data = await db.document.upsert(
                where={"id": doc_id},
                data=upsert_data
            )
            print(f"Upserted a Document record with ID: {upserted_data.id}")

    return png_bytes_b64


def write_batch_jsonl_file(batch_lines: list[str], filename: str) -> None:
    """
    Write the given list of JSON lines to `filename` with newlines in between.
    """
    with open(filename, "w", encoding="utf-8") as f:
        print("\n".join(batch_lines), file=f)


def create_and_submit_batch(batch_file: str) -> str:
    """
    Upload the JSONL batch_file to OpenAI, create a Batch, and return batch_id.
    """
    print("Uploading batch input file to OpenAI...")
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
    print('created a batch with id: ', created_batch.id)
    return batch_id


def poll_batch(batch_id: str, sleep_seconds: int = 10) -> Batch:
    """
    Poll the Batch endpoint for completion every `sleep_seconds`.
    Returns the final batch status JSON once completed or failed.
    """
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"Current batch status: {status}")
        if status in ["completed", "failed", "canceled", "expired"]:
            return batch
        time.sleep(sleep_seconds)

# ----------------------------------------------------------------------------
# 7. ORCHESTRATION (MAIN)
# ----------------------------------------------------------------------------
async def main():
    # 7.1 Connect DB & load environment variables
    db = await connect_db()
    input_json_path = "json_files/input_data_example.json"

    # 7.2 Read input data
    data: InputDataType = load_input_data(input_json_path)

    # 7.3 For each document, process PDF->PNG, store in DB, build user content
    batch_lines = []
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
        batch_lines.append(line)

    batch_dir = 'batches'

    # 7.4 Write the .jsonl file
    batch_input_file = os.path.join(batch_dir, "batchinput.jsonl")
    write_batch_jsonl_file(batch_lines, batch_input_file)

    # (Below steps can remain optional / commented out if you just want the .jsonl)
    # 7.5 Create the Batch, poll, and store results
    batch_id = create_and_submit_batch(batch_input_file)
    batch = poll_batch(batch_id, sleep_seconds=10)
    api_call_result_dir = 'api_call_results'
    output_filename = os.path.join(api_call_result_dir, f"out-{batch_id}.jsonl")
    await process_batch_output(db, batch, output_filename)

    # 7.6 Disconnect
    await disconnect_db(db)


if __name__ == "__main__":
    asyncio.run(main())
