#!/usr/bin/env python3

"""
build-finetune-dataset.py
-------------------------
Fetches data from your PostgreSQL (via Prisma) and creates a .jsonl file
suitable for OpenAI fine-tuning using the chat format.

Usage:
  ./build-finetune-dataset.py
"""

import asyncio
import json
import os
import glob
from typing import List, Literal
from fine_tuning_utils.system_prompt import system_prompt
from prisma_utils.prisma_utils import disconnect_db, get_prisma_db
from pydantic import BaseModel
from fine_tuning_utils.dataset_statistics import calculate_billing_tokens, estimate_n_epochs, print_billing_info
from dotenv import load_dotenv

from submit_finetune_job import N_EPOCHS

load_dotenv()


class Message(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str

    def to_json(self) -> str:
        return self.model_dump_json()


async def build_finetune_dataset(include_ids: List[str]) -> List[dict]:
    """
    Fetches data from the database, processes it, and returns a list of fine-tuning dataset lines.
    Only documents whose id is in include_ids are retrieved.
    """
    prisma = await get_prisma_db()

    documents = await prisma.document.find_many(
        where={
            "id": {
                "in": include_ids
            }
        },
        include={
            "messages": True,
            "DocumentTranscription": True,
        },
    )
    print(f"Found {len(documents)} documents...")

    await disconnect_db(prisma)
    exit(0)

    lines = []

    for doc in documents:
        if not doc.DocumentTranscription:
            print(f'WARN -- no DocumentTranscription for {doc.id}')
            continue

        sorted_transcriptions = sorted(
            doc.DocumentTranscription,
            key=lambda t: t.version,
            reverse=True
        )
        document_representation = sorted_transcriptions[0].document_representation

        if not doc.messages:
            print(f'WARN -- no messages for {doc.id}')
            continue

        fine_tuning_entries: List[Message] = [
            Message(role=m.role, content=m.content) for m in doc.messages
        ]

        if not fine_tuning_entries:
            print(f'WARN -- no assistant messages for {doc.id}')
            continue

        new_samples = [
            {
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': document_representation},
                    *[message.model_dump() for message in conversation_slice]
                ]
            }
            for conversation_slice in [fine_tuning_entries]
        ]

        print(f'{doc.id}  About to add {len(new_samples)} new samples...')
        lines.extend(new_samples)

    await disconnect_db(prisma)
    return lines


def save_finetune_dataset(lines: List[dict], output_file: str) -> None:
    """
    Saves the fine-tuning dataset to a JSONL file.
    """
    with open(output_file, "w", encoding="utf-8") as f_out:
        for line in lines:
            f_out.write(json.dumps(line) + "\n")
    print(f"Successfully created {output_file}")


def calculate_and_print_statistics(lines: List[dict]) -> None:
    """
    Calculates and prints statistics for the fine-tuning dataset.
    """
    convo_lens = [
        sum(len(message['content']) for message in entry['messages'])
        for entry in lines
    ]
    n_billing_tokens = calculate_billing_tokens(convo_lens)
    n_epochs = estimate_n_epochs(len(lines), n_epochs=N_EPOCHS)

    print_billing_info(n_epochs, n_billing_tokens, 'gpt-4o-mini')


def gather_include_ids(directory: str) -> List[str]:
    """
    Reads all .jsonl files in the specified directory, collects 'custom_id' values
    from each line, and returns them as a list.
    """
    include_ids = []
    for file_path in glob.glob(os.path.join(directory, "*.jsonl")):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "custom_id" in data:
                        include_ids.append(data["custom_id"])
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON in {file_path}: {line}")
    return include_ids


async def main(out_file: str):
    # Collect all custom IDs from ../data/api_call_results/*.jsonl
    custom_ids = gather_include_ids("../data/api_call_results")
    print(
        f"Found {len(custom_ids)} custom IDs in ../data/api_call_results/*.jsonl"
    )

    # Pass those IDs into the build_finetune_dataset function
    lines = await build_finetune_dataset(include_ids=custom_ids)

    print(f'Total lines: {len(lines)}')

    # Save to file
    save_finetune_dataset(lines, output_file=out_file)

    # Print statistics
    calculate_and_print_statistics(lines)


if __name__ == "__main__":
    OUTPUT_FILE = "../data/fine_tune_data/second_try_only_800plus_response_tokens.jsonl"
    asyncio.run(main(OUTPUT_FILE))
