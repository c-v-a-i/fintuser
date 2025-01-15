import asyncio
import base64
import json
import os
from io import BytesIO
from typing import Dict, List
from typing import Any

from chat_data_transform_utils.batch_api_utils import chunk_batch_lines, write_batch_jsonl_file, create_and_submit_batch, poll_batches_until_done, process_completed_batches
from chat_data_transform_utils.system_prompt import system_prompt
from pdf2image import convert_from_path
from prisma import Prisma
from prisma.types import (
    DocumentUpsertInput,
)
from chat_data_transform_utils.build_batch_line import build_batch_line
from prisma_utils.prisma_utils import get_prisma_db, disconnect_db

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
    db = await get_prisma_db()

    # We assume this JSON file contains the "doc_id -> {...}" mapping
    input_json_path = "../data/json_files/pdf_children_texts_ivGBh.json"

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

    ## TODO: each batch should be no more than 90k tokens for gpt-4o.
    # 7.3.4 Split the lines into multiple chunked lists if the size is too large
    chunked_batches = chunk_batch_lines(all_batch_lines, max_batch_file_size_bytes=50 * 1024 * 1024)
    batch_dir = '../data/batches'
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
    api_call_result_dir = '../data/api_call_results'
    await process_completed_batches(db, final_batches, api_call_result_dir)

    # 7.7 Disconnect
    await disconnect_db(db)


async def after_batches_completed(batches_ids: List[str]):
    """
    After all batches are completed, process the output files.
    """
    db = await get_prisma_db()

    completed_batches = poll_batches_until_done(batches_ids, sleep_seconds=10)

    api_call_result_dir = '../data/api_call_results'
    await process_completed_batches(db, completed_batches, api_call_result_dir)

    # 7.7 Disconnect
    await disconnect_db(db)


if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(after_batches_completed(
        ['batch_6786c608481c819098fc8f1453a5bcbb', 'batch_6786c5ec39248190b133b5ca29a1c441', 'batch_6786c5f5eb4081909a4af1d2d87703bb', 'batch_6786c5fe86948190ad6c66f73bd5b770', 'batch_6786c60edbdc8190964604558d4816f8']
    ))
