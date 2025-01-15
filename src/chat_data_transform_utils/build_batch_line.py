import json
import openai.lib._parsing as parsing
from .response_schema import GPTOutputSchema


def build_batch_line(
        doc_id: str,
        review_conversation: str,
        png_bytes_b64: str,
        system_prompt: str,
        model: str
) -> str:
    user_message_content = [
        {
            'type': 'image_url',
            'image_url': {
                'url': f"data:image/png;base64,{png_bytes_b64}"
            }
        },
        {
            'type': 'text',
            'text': review_conversation
        },
    ]

    response_format = parsing.type_to_response_format_param(
        GPTOutputSchema
    )

    batch_line_dict = {
        "custom_id": doc_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message_content},
            ],
            "response_format": response_format,
            "max_tokens": 8000,
        },
    }
    return json.dumps(batch_line_dict)
