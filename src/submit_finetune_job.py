"""
submit_finetune_job.py
----------------------
Submits your JSONL files to OpenAI for fine-tuning.

Usage:
  export OPENAI_API_KEY="sk-..."
  ./submit_finetune_job.py <training_dataset.jsonl> <validation_dataset.jsonl>
"""

import os
import json
import sys
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Hyperparameters(BaseModel):
    n_epochs: int
    batch_size: int
    learning_rate_multiplier: float = None


HYPERPARAMETERS = Hyperparameters(
    n_epochs=2,
    batch_size=8,
    # learning_rate_multiplier=1
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini-2024-07-18"


def upload_file(file_path: str) -> str:
    """
    Uploads a file to OpenAI for fine-tuning purposes.
    Returns the file ID.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set.")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'application/jsonl')}
        data = {'purpose': 'fine-tune'}

        response = requests.post(
            f"{OPENAI_API_BASE}/files",
            headers=headers,
            files=files,
            data=data
        )

    if response.status_code != 200:
        raise Exception(f"Failed to upload file: {response.text}")
    resp_json = response.json()
    return resp_json["id"]


def create_finetune_job(training_file_id: str, validation_file_id: str, model: str, hyperparameters: Hyperparameters):
    """
    Creates a fine-tuning job using the specified uploaded training and validation files.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "training_file": training_file_id,
        "validation_file": validation_file_id,
        "model": model,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": hyperparameters.model_dump(exclude_none=True)
            }
        }
    }

    response = requests.post(
        f"{OPENAI_API_BASE}/fine_tuning/jobs",
        headers=headers,
        json=data
    )
    if response.status_code != 200:
        raise Exception(f"Failed to create fine-tuning job: {response.text}")
    return response.json()


def list_finetune_jobs(limit=10):
    """
    Lists your organization's fine-tuning jobs, limited to the specified number.
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    response = requests.get(
        f"{OPENAI_API_BASE}/fine_tuning/jobs?limit={limit}",
        headers=headers
    )
    if response.status_code != 200:
        raise Exception(f"Failed to list fine-tuning jobs: {response.text}")
    return response.json()


def run_finetuning_job(training_dataset: str, validation_dataset: str):
    """
    Runs the fine-tuning job submission process by uploading the training and validation files,
    creating a fine-tuning job, and listing current fine-tuning jobs.
    """
    # Verify both files exist
    if not os.path.isfile(training_dataset):
        raise FileNotFoundError(f"Training dataset file '{training_dataset}' does not exist.")
    if not os.path.isfile(validation_dataset):
        raise FileNotFoundError(f"Validation dataset file '{validation_dataset}' does not exist.")

    # Upload training file
    print(f"Uploading training file: {training_dataset}...")
    training_file_id = upload_file(training_dataset)
    print(f"Training file uploaded successfully: {training_file_id}")

    # Upload validation file
    print(f"Uploading validation file: {validation_dataset}...")
    validation_file_id = upload_file(validation_dataset)
    print(f"Validation file uploaded successfully: {validation_file_id}")

    # Create fine-tuning job
    print(f"Creating fine-tuning job for model={MODEL} with hyperparameters: {HYPERPARAMETERS.model_dump()}...")
    job_response = create_finetune_job(
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        model=MODEL,
        hyperparameters=HYPERPARAMETERS
    )
    print("Fine-tuning job created successfully.")
    print("Job details:")
    print(json.dumps(job_response, indent=2))

    # List fine-tuning jobs
    print("\nListing current fine-tuning jobs...")
    jobs = list_finetune_jobs()
    print(json.dumps(jobs, indent=2))


def main():
    if not OPENAI_API_KEY:
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Ensure both training and validation dataset arguments are provided
    if len(sys.argv) < 3:
        print("Usage: ./submit_finetune_job.py <training_dataset.jsonl> <validation_dataset.jsonl>")
        sys.exit(1)

    training_dataset = sys.argv[1]
    validation_dataset = sys.argv[2]

    try:
        run_finetuning_job(training_dataset, validation_dataset)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
