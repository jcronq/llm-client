from enum import Enum
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class Role(Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"
