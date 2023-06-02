from typing import TypeAlias, Iterable
from datetime import datetime
import operator

from pydantic import BaseModel
import numpy as np
import sqlite3

from llm_client.agent.prompt import Prompt
from llm_client.llm_utils import create_embedding_with_ada
from llm_client.types.openai import Role, Message

from llm_client.agent.memory.sql_backed_memory_objects import SqlMessage, SqlInteraction, Vector
from llm_client.agent.memory.message_store import MessageStore
from llm_client.agent.memory.interaction_store import InteractionStore
from llm_client.agent.memory.interaction import Interaction
from llm_client.agent.memory.remembered_interaction import RememberedInteraction


class Memory:
    def __init__(self, database_file: str):
        self.message_store = MessageStore()
        self.interaction_store = InteractionStore(self.message_store)
        self.db_path = database_file
        self._create_db_tables()
        self.load()

    def load(self):
        with sqlite3.connect(self.db_path) as conn:
            sql_messages: list[SqlMessage] = SqlMessage.load_all(conn)
            for message in sql_messages:
                self.message_store.add_message(message)
            sql_interactions: list[SqlInteraction] = SqlInteraction.load_all(conn)
            for interaction in sql_interactions:
                self.interaction_store.add_interaction(interaction)

    def _create_db_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            for table in SqlMessage.sql_tables():
                cursor = conn.cursor()
                cursor.execute(table)

            for table in SqlInteraction.sql_tables():
                cursor = conn.cursor()
                cursor.execute(table)
            conn.commit()

    def get_message_id(self, role: Role, text: str):
        return self.get_message(role, text).uid

    def get_message(self, role: Role, text: str):
        sqlmessage: SqlMessage = self.message_store.lookup_by_text(role, text)
        if sqlmessage is None:
            sqlmessage = SqlMessage(role=role.value, text=text, embedding=Vector(data=create_embedding_with_ada(text)))
        return sqlmessage

    def add_interaction(self, prompt: Prompt, reply: str):
        user_message_id = self._save_message(prompt.user_message)
        response_message_id = self._save_message(self.get_message(Role.Assistant, reply))
        system_message_ids = [self._save_message(message) for message in prompt.system_messages]
        relevant_interaction_ids = [
            remembered_interaction.uid for remembered_interaction in prompt.relevant_interactions
        ]
        recent_interaction_ids = [remembered_interaction.uid for remembered_interaction in prompt.recent_interactions]
        interaction = SqlInteraction(
            created_at=datetime.utcnow(),
            user_message_id=user_message_id,
            response_message_id=response_message_id,
            system_message_ids=system_message_ids,
            relevant_interaction_ids=relevant_interaction_ids,
            recent_interaction_ids=recent_interaction_ids,
        )
        self._save_interaction(interaction)

    def get_remembered_interaction_from_id(self, interaction_id):
        interaction = self.interaction_store.lookup_by_id(interaction_id)
        RememberedInteraction(
            self.message_store.lookup_by_id(interaction.user_message_id),
            self.message_store.lookup_by_id(interaction.response_message_id),
        )

    def render_prior_interaction(self, interaction: SqlInteraction) -> RememberedInteraction:
        # system_msgs = [self.message_store.lookup_by_id(message_id) for message_id in interaction.system_message_ids]
        # relevant_rememberances = [
        #     self.get_remembered_interaction_from_id(interaction_id)
        #     for interaction_id in interaction.relevant_message_ids
        # ]
        # recent_rememberances = [
        #     self.get_remembered_interaction_from_id(interaction_id)
        #     for interaction_id in interaction.relevant_message_ids
        # ]

        user_message = self.message_store.lookup_by_id(interaction.user_message_id)
        return RememberedInteraction(
            uid=interaction.uid,
            created_at=user_message.created_at,
            user_message=user_message.text,
            response_message=self.message_store.lookup_by_id(interaction.response_message_id).text,
        )

    def _save_message(self, message: SqlMessage) -> str:
        if self.message_store.lookup_by_id(message.uid) is None:
            with sqlite3.connect(self.db_path) as conn:
                message.save_to_sql(conn)
            self.message_store.add_message(message)
        return message.uid

    def _save_interaction(self, interaction: SqlMessage) -> str:
        if self.interaction_store.lookup_by_id(interaction.uid) is None:
            with sqlite3.connect(self.db_path) as conn:
                interaction.save_to_sql(conn)
        self.interaction_store.add_interaction(interaction)

    @property
    def messages(self) -> list[SqlMessage]:
        return self.message_store.values()

    @property
    def user_messages(self) -> list[SqlMessage]:
        return [message for message in self.message_store.values() if message.role == Role.User]

    @property
    def non_system_messages(self) -> list[SqlMessage]:
        return [message for message in self.message_store.values() if message.role != Role.System]

    @property
    def interactions(self) -> list[SqlInteraction]:
        return self.interaction_store.values()

    def k_most_similar(self, text: str, k: int) -> tuple[list[tuple[Prompt, Prompt]], list[Prompt]]:
        """ "
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        return self.k_most_similar_interactions(text, self.non_system_messages, k)

    def k_most_similar_inputs(self, text: str, k: int) -> list[Interaction]:
        return self.k_most_similar_interactions(text, self.user_messages, k)

    def k_most_similar_messages(self, text: str, messages: list[SqlMessage], k: int) -> list[SqlMessage]:
        if len(messages) == 0:
            return []
        test_embedding: list[float] = create_embedding_with_ada(text)
        embeddings: list[list[float]] = [message.embedding.data for message in messages]

        # What if scores were weighted by how frequent they appear as well?  What would that look like?
        scores = np.dot(embeddings, test_embedding)

        return [messages[idx] for idx in np.argsort(scores)[-k:][::-1]]

    def k_most_similar_interactions(self, text: str, messages: list[SqlMessage], k: int) -> list[SqlInteraction]:
        return [
            self.interaction_store.lookup_by_user_msg_id(message.uid)
            for message in self.k_most_similar_messages(text, messages, k)
        ]

    def k_most_recent(self, k: int):
        return self.interaction_store.time_sorted_interactions()[:k:][::-1]
