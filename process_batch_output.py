import json
from prisma.enums import Role
from prisma import Prisma
from openai.types import Batch
from openai_client import client
from response_schema import GPTOutputSchema


async def save_response_to_db(text_response: str, db: Prisma):
    """
    Parse JSONL lines from `text_response` and save them to the database.
    """
    # Split the downloaded text by lines
    lines = text_response.splitlines()

    for line in lines:
        line = line.strip()
        # Skip empty lines or lines that are single braces
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # If a line is not valid JSON, decide how you want to handle it:
            # skip or log an error. Here we skip for simplicity:
            print(f'Could not parse line "{line}"')
            continue

        # Extract metadata
        document_id: str | None = data.get("custom_id")
        if not document_id:
            # If there's no suitable ID, generate one or skip
            # Example fallback:
            print(f'Could not get custom_id (doc_id) for line "{line}"')
            continue

        # Check if we have an error
        error_obj = data.get("error")
        if error_obj:
            # You might want to log this or save in a special table
            print(f"Error found in batch item with id {document_id}: {error_obj}")
            continue

        # Get the response structure
        response = data.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            continue

        # The assistant's message content is stored in choices[0]["message"]["content"]
        raw_content = choices[0]["message"]["content"]

        try:
            gpt_data = GPTOutputSchema.model_validate_json(raw_content)
        except Exception as e:
            print(f"Failed to parse GPT output for document {document_id}: {e}")
            continue

        for msg in gpt_data.conversation_translation:
            prisma_role: Role = Role.assistant if msg.type == "assistant" else Role.user

            await db.documentmessage.create(
                {
                    "role": prisma_role,
                    "content": msg.content,
                    "documentId": document_id
                }
            )

        await db.documenttranscription.create(
            {
                "version": 1,
                "document_representation": gpt_data.document_representation,
                "documentId": document_id
            }
        )

        print(f"Saved document {document_id} with {len(gpt_data.conversation_translation)} messages")



async def process_batch_output(
    db: Prisma,
    batch:  Batch,
    output_filename
) -> None:
    """
    If the batch is completed, download the results, parse them,
    and store them in the DB.
    """
    batch_status = batch.status
    if batch_status != "completed":
        print(f"Batch {batch.id} not completed successfully. Final status: {batch_status}")
        print(f'batch: {batch}')
        return

    error_file_id = batch.error_file_id

    if error_file_id:
        print("Error file presents.")
        content = client.files.content(batch.error_file_id)
        print(f"Error file content: {content.text}")
        return

    output_file_id = batch.output_file_id
    print(f"Batch {batch.id} completed. Output file: {output_file_id}")

    # Retrieve and download results
    result_file = client.files.content(output_file_id)
    batch_output_text = result_file.text
    print(f'Batch output text: {batch_output_text}')

    text_response = result_file.text
    with open(output_filename, "w") as f:
        print(text_response, file=f)

    await save_response_to_db(text_response, db)
