# fintuser

[Notion documentation](https://adaptive-omelet-0bd.notion.site/fine-tuning-data-service-FTDS-16854548733a8020b2ecfb9fbb560a29?pvs=4)


## Introduction

I want to have an LLM-powered "CV reviewer" as a part of my CVAI project.

I have [the following chat in Telegram](https://t.me/resume_review), where engineers review the CVs of other engineers.

The idea is to use the reviews from this chat for fine-tuning an LLM.


## Description
A repository with the python scripts needed for creating a dataset from fine-tuning and submitting all the jobs.


- [create-message-chains.py](src/create_message_chains.py) takes care of transformation unstructured data into `ProcessedMessages` type. This will craete message chains and store them. They are later converted into english and then into the finetuning dataset format. 
- [transform_chat_data.py](src/transform_chat_data.py) takes the output of `create-message-chains.py`, and uses gpt-4o-mini to translate the text into english and add a yaml representation of PDF documents. So we can fine-tune both on raw documents and the transcriptions of the documents.
- [build-finetune-dataset](src/build_finetune_dataset.py) creates a dataset for fine-tuning given the data in the database
- [submit-finetune-job](src/submit_finetune_job.py) uploads a file with a fine-tuning dataset and submits a job
- [save_out_dir_to_db](src/save_out_dir_to_db.py) gets output of batched jobs and saves it to the database


## file structure

- ./data - json data + chat documents
- [./data/raw_chat_data/](./data/raw_chat_data) - json from messages from telegram chat
- [./data/json_files/](./data/json_files) - dataset in the format "doc_id" -> messages + filepath. The messages are in russian, and they need to be translated. Each file is a complete dataset. They differ by the number of entries and the pre-processing thingies. The format is the same though.
- [./data/files/](./data/files) - PDFs. Ideally, this directory should be deleted and database dump should be enough.
- [./data/batches/](./data/batches) - batchs ready to be submitted for translation: russian -> english.
- [./data/fine_tune_data/](./data/fine_tune_data) - translated text suitable for fine-tuning
