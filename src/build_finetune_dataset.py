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


def collapse_messages(messages: List[Message]) -> List[Message]:
    new_messages = []
    prev_role = messages[0].role
    content_acc = messages[0].content

    # Start from the second message to compare roles
    for message in messages[1:]:
        if message.role != prev_role:
            # Append the previous chunk before switching roles
            new_messages.append(
                Message(
                    role=prev_role,
                    content=content_acc,
                )
            )
            # Reset accumulators
            prev_role = message.role
            content_acc = message.content
        else:
            content_acc += f'\n{message.content}'

    # Append whatever remains
    new_messages.append(
        Message(
            role=prev_role,
            content=content_acc,
        )
    )

    return new_messages


def split_messages(messages: List[Message]) -> List[List[Message]]:
    samples = []
    # Use `+ 1` so that 2 messages produce exactly 1 sample containing both
    for i in range(1, (len(messages) // 2) + 1):
        samples.append(messages[: i * 2])
    return samples


async def build_finetune_dataset() -> List[dict]:
    """
    Fetches data from the database, processes it, and returns a list of fine-tuning dataset lines.
    """
    prisma = await get_prisma_db()

    # Fetch all documents, including relations
    documents = await prisma.document.find_many(
        include={
            "messages": True,
            "DocumentTranscription": True,
        },
    )
    print(f"Found {len(documents)} documents...")

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

        # Prepend document representation
        starter_message: Message = Message(
            role='user',
            content=document_representation
        )

        document_conversation: List[Message] = [starter_message] + [
            Message(role=m.role, content=m.content) for m in doc.messages
        ]

        # Combine repeating messages together
        collapsed_messages = collapse_messages(document_conversation)

        # Split messages into multiple samples
        fine_tuning_entries = split_messages(collapsed_messages)

        new_samples = list(map(lambda conversation_slice: {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                *[message.model_dump() for message in conversation_slice]
            ]
        }, fine_tuning_entries))

        print(f'{doc.id}  About to add {len(new_samples)} new samples...')
        # Add system prompt and serialize
        lines.extend(
            new_samples
        )

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


async def main(out_file: str):
    # Build the dataset
    lines = await build_finetune_dataset()

    print(f'Total lines: {len(lines)}')

    # Save to file
    save_finetune_dataset(lines, output_file=out_file)

    # Print statistics
    calculate_and_print_statistics(lines)


if __name__ == "__main__":
    OUTPUT_FILE = "../fine_tune_data/finetune_dataset.jsonl"

    asyncio.run(main(OUTPUT_FILE))
