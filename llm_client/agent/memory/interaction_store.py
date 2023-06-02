from typing import Optional
from llm_client.agent.memory.sql_backed_memory_objects import SqlInteraction
from llm_client.agent.memory.message_store import MessageStore
from llm_client.agent.memory.interaction import Interaction


class InteractionStore:
    def __init__(self, message_store: MessageStore):
        self.message_store = message_store
        self.user_msg_id_to_interaction: dict[int, SqlInteraction] = {}
        self.id_to_interaction: dict[str, SqlInteraction] = {}

    def lookup_by_text(self, text: str) -> Optional[SqlInteraction]:
        user_message_id = self.message_store.lookup_by_text(text)
        return self.user_msg_id_to_interaction.get(user_message_id, None)

    def lookup_by_user_msg_id(self, user_message_id: str) -> Optional[SqlInteraction]:
        return self.user_msg_id_to_interaction.get(user_message_id, None)

    def lookup_by_id(self, interaction_id: str):
        return self.id_to_interaction.get(interaction_id, None)

    def add_interaction(self, interaction: SqlInteraction):
        self.user_msg_id_to_interaction[interaction.user_message_id] = interaction
        self.id_to_interaction[interaction.uid] = interaction

    def values(self) -> list[SqlInteraction]:
        return list(self.id_to_interaction.values())

    def time_sorted_interactions(self) -> list[SqlInteraction]:
        return list(sorted(self.id_to_interaction.values(), key=lambda interaction: interaction.created_at))
