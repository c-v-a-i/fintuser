import asyncio
import sys
import os

from chat_data_transform_utils.process_batch_output import save_response_to_db
from prisma_utils.prisma_utils import get_prisma_db, disconnect_db


async def main(directory: str) -> None:
    db = await get_prisma_db()

    # Ensure directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        await db.disconnect()
        return

    # Process all *.json files
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text_response = f.read()

            print(f'Processing {filename}...')

            # Call the existing async function to parse and save data
            await save_response_to_db(text_response, db)

    await disconnect_db(db)

if __name__ == '__main__':
    # Allow passing a directory as a command-line argument or default to ./outputs
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '../api_call_results'
    asyncio.run(main(target_dir))