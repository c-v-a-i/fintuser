import time
from typing import List, Tuple
from openai.types import Batch
from prisma import Prisma
import os
from process_batch_output import process_batch_output, save_response_to_db
from openai_client.openai_client import client


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
            text_response = await process_batch_output(batch_obj, output_filename)
            if text_response:
                await save_response_to_db(text_response, db)
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
