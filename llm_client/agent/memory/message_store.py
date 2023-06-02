from typing import Optional
from llm_client.agent.memory.sql_backed_memory_objects import SqlMessage
from llm_client.types.openai import Role


class MessageStore:
    def __init__(self):
        self.hash_to_message: dict[Role, dict[int, SqlMessage]] = {Role.System: {}, Role.User: {}, Role.Assistant: {}}
        self.id_to_message: dict[str, SqlMessage] = {}

    def lookup_by_text(self, role: Role, text: str) -> Optional[SqlMessage]:
        return self.hash_to_message[role].get(hash(text), None)

    def lookup_by_id(self, message_id: str) -> Optional[SqlMessage]:
        return self.id_to_message.get(message_id, None)

    def add_message(self, message: SqlMessage):
        self.hash_to_message[message.role][hash(message)] = message
        self.id_to_message[message.uid] = message
    
    def values(self) -> list[SqlMessage]:
        return list(self.id_to_message.values())
    