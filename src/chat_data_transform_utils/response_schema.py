from typing import List
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class ConversationItem(BaseModel):
    type: Literal["assistant", "user"]
    content: str


class GPTOutputSchema(BaseModel):
    document_representation: str = Field(..., description="YAML representation of the CV outputted as a string")
    conversation_translation: List[ConversationItem]
