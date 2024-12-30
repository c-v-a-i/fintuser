# fintuser

[Notion documentation](https://adaptive-omelet-0bd.notion.site/fine-tuning-data-service-FTDS-16854548733a8020b2ecfb9fbb560a29?pvs=4)


## Introduction

I want to have an LLM-powered "CV reviewer" as a part of my CVAI project.

I have [the following chat in Telegram](https://t.me/resume_review), where engineers review the CVs of other engineers.

The idea is to use the reviews from this chat for fine-tuning an LLM.


## Description

We have exported the chat history in  [`result.json`](json_files/result.json).

The messages are in russian, and they're unprocessed.

In order to have meaningful results from the fine-tuning, we need to pre-process the messages so the dataset has the following format:
```typescript
type ProcessedMessages = {
  [id in string]: {
      messages: OpenaiMessage; // { type: 'user' | 'assistant'; content: string }
      pdf_filename: string;
  }
}
```

I filtered out CVs with the names that don't make much sense. Like `xxxx.pdf` or `test.pdf` or `resume (79).pdf`.

There are no script doing this job so far in the repo.

- [create-message-chains.py](src/create_message_chains.py) takes care of transformation unstructured data into `ProcessedMessages` type.
- [transform_chat_data.py](src/transform_chat_data.py) takes the output of `create-message-chains.py`, and uses gpt-4o-mini to translate the text into english and add a yaml representation of PDF documents. So we can fine-tune both on raw documents and the transcriptions of the documents.
