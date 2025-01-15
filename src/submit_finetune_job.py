#!/usr/bin/env python3

"""
submit_finetune_job.py
----------------------
Submits your JSONL file to OpenAI for fine-tuning.

Usage:
  export OPENAI_API_KEY="sk-..."
  ./submit_finetune_job.py <training_dataset.jsonl>
"""

import os
import json
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

N_EPOCHS = 8
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

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), 'application/jsonl')
    }
    data = {
        'purpose': 'fine-tune'
    }

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


def create_finetune_job(file_id: str, model: str, n_epochs: int):
    """
    Creates a fine-tuning job using the specified uploaded file.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "training_file": file_id,
        "model": model,
        "method": {
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": n_epochs,
                    "batch_size": 4,
                }
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
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    response = requests.get(
        f"{OPENAI_API_BASE}/fine_tuning/jobs?limit={limit}",
        headers=headers
    )
    if response.status_code != 200:
        raise Exception(f"Failed to list fine-tuning jobs: {response.text}")
    return response.json()


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Ensure a training dataset argument is provided
    if len(sys.argv) < 2:
        print("Usage: ./submit_finetune_job.py <training_dataset.jsonl>")
        sys.exit(1)

    TRAINING_DATASET = sys.argv[1]

    if not os.path.isfile(TRAINING_DATASET):
        print(f"Error: File '{TRAINING_DATASET}' does not exist.")
        sys.exit(1)

    # 1) Upload the fine-tuning dataset
    print(f"Uploading {TRAINING_DATASET} to OpenAI...")
    try:
        file_id = upload_file(TRAINING_DATASET)
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)
    print(f"File uploaded successfully: {file_id}")

    # 2) Create a fine-tuning job
    print(f"Creating fine-tuning job for model={MODEL}, epochs={N_EPOCHS}...")
    try:
        job_response = create_finetune_job(file_id=file_id, model=MODEL, n_epochs=N_EPOCHS)
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        sys.exit(1)

    print("Fine-tuning job created successfully.")
    print("Job details:")
    print(json.dumps(job_response, indent=2))

    # 3) (Optional) List fine-tuning jobs
    print("\nListing current fine-tuning jobs...")
    try:
        jobs = list_finetune_jobs()
        print(json.dumps(jobs, indent=2))
    except Exception as e:
        print(f"Error listing fine-tuning jobs: {e}")
