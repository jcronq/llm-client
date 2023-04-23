from typing import TypedDict
from enum import Enum


class Message(TypedDict):
    role: str
    content: str


class Role(Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"
