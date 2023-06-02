from typing import Optional
from dateutil.parser import parse as parse_date_string
import struct
from uuid import uuid4
from pydantic import BaseModel, Field
from datetime import datetime
import sqlite3

from llm_client.types.openai import Role


def vector_to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def blob_to_vector(blob_data: bytes) -> list[float]:
    num_floats = len(blob_data) // 4
    return list(struct.unpack(f"{num_floats}f", blob_data))


class Vector(BaseModel):
    data: list[float]

    @property
    def blob(self) -> bytes:
        return vector_to_blob(self.data)

    @classmethod
    def from_blob(cls, blob_data: bytes) -> "Vector":
        return cls(data=blob_to_vector(blob_data))


class SqlMessage(BaseModel):
    role: Role
    text: str
    embedding: Vector
    uid: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())

    def __hash__(self):
        return hash(self.text)

    @classmethod
    def load_from_sql(cls, conn: sqlite3.Connection, message_uid: str) -> Optional["SqlMessage"]:
        # Retrieve the row from the messages table with the given message_id
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, timestamp, role, content, embedding FROM messages WHERE id = ?",
            (message_uid,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        uid, created_at, role_str, text, message_embedding = row
        embedding = Vector.from_blob(message_embedding)
        return SqlMessage(parse_date_string(created_at), Role(role_str), text, embedding, uid)

    @classmethod
    def load_all(cls, conn: sqlite3.Connection) -> list["SqlMessage"]:
        # Retrieve the row from the messages table with the given message_id
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, timestamp, role, content, embedding FROM messages",
        )
        rows = cursor.fetchall()
        messages = []
        for row in rows:
            uid, created_at, role_str, text, message_embedding = row
            embedding = Vector.from_blob(message_embedding)
            messages.append(
                cls(
                    created_at=parse_date_string(created_at),
                    role=Role(role_str),
                    text=text,
                    embedding=embedding,
                    uid=uid,
                )
            )
        return messages

    def save_to_sql(self, conn: sqlite3.Connection):
        cursor = conn.cursor()
        # Insert a new row into the messages table
        cursor.execute(
            "INSERT INTO messages (id, timestamp, role, content, embedding) VALUES (?, ?, ?, ?, ?)",
            (self.uid, self.created_at, self.role.value, self.text, self.embedding.blob),
        )

    @classmethod
    def sql_tables(cls):
        return [
            (
                "CREATE TABLE IF NOT EXISTS messages ("
                "   id TEXT PRIMARY KEY,"
                "   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
                "   role TEXT NOT NULL,"
                "   content TEXT NOT NULL,"
                "   embedding BLOB NOT NULL"
                ")"
            )
        ]


class SqlInteraction(BaseModel):
    created_at: datetime
    user_message_id: str
    response_message_id: str
    system_message_ids: list[str]
    relevant_interaction_ids: list[str]
    recent_interaction_ids: list[str]
    uid: str = Field(default_factory=lambda: uuid4().hex)

    @classmethod
    def load_all(cls, conn: sqlite3.Connection) -> list["SqlInteraction"]:
        # Retrieve the row from the messages table with the given message_id
        cursor = conn.cursor()

        cursor.execute(
            (
                "  SELECT"
                "    i.created_at,"
                "    i.id AS uid,"
                "    i.user_message_id,"
                "    i.response_message_id,"
                "    GROUP_CONCAT(ism.system_message_id) AS system_message_ids,"
                "    GROUP_CONCAT(irm.related_interaction_id) AS relevant_interaction_ids,"
                "    GROUP_CONCAT(rrm.related_interaction_id) AS recent_interaction_ids"
                "  FROM"
                "    interactions i"
                "    LEFT JOIN interaction_system_messages ism ON i.id = ism.interaction_id"
                "    LEFT JOIN interaction_relevant_interactions irm ON i.id = irm.interaction_id"
                "    LEFT JOIN interaction_recent_interactions rrm ON i.id = rrm.interaction_id"
                "  GROUP BY"
                "    i.id;"
            )
        )

        interactions = []
        rows = cursor.fetchall()
        for row in rows:
            (
                created_at,
                uid,
                user_message_id,
                response_message_id,
                system_message_ids_str,
                relevant_interaction_ids,
                recent_interaction_ids,
            ) = row

            interactions.append(
                cls(
                    created_at=created_at,
                    uid=uid,
                    user_message_id=user_message_id,
                    response_message_id=response_message_id,
                    system_message_ids=system_message_ids_str.split(",") if system_message_ids_str is not None else [],
                    relevant_interaction_ids=relevant_interaction_ids.split(",")
                    if relevant_interaction_ids is not None
                    else [],
                    recent_interaction_ids=recent_interaction_ids.split(",")
                    if system_message_ids_str is not None
                    else [],
                )
            )
        return interactions

    @classmethod
    def load_from_sql(cls, conn: sqlite3.Connection, interaction_uid: str) -> Optional["SqlInteraction"]:
        # Retrieve the row from the messages table with the given message_id
        cursor = conn.cursor()

        cursor.execute(
            "SELECT created_at, id, user_message_id, response_message_id FROM messages WHERE id = ?",
            (interaction_uid,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        created_at, uid, user_message_id, response_message_id = row

        cursor.execute(
            "SELECT system_message_id FROM system_prompt where interaction_id = ?",
            (interaction_uid),
        )
        # Retrieve the related prompt IDs from the related_messages table
        system_message_ids = [row[0] for row in cursor.fetchall()]

        cursor.execute(
            "SELECT related_interaction_id FROM interaction_relevant_interactions where interaction_id = ?",
            (interaction_uid),
        )
        # Retrieve the related prompt IDs from the related_messages table
        relevant_interaction_ids = [row[0] for row in cursor.fetchall()]

        cursor.execute(
            "SELECT related_interaction_id FROM interaction_recent_interactions where interaction_id = ?;",
            (interaction_uid),
        )
        # Retrieve the related prompt IDs from the related_messages table
        recent_interaction_ids = [row[0] for row in cursor.fetchall()]

        return cls(
            created_at,
            uid,
            user_message_id,
            response_message_id,
            system_message_ids,
            relevant_interaction_ids,
            recent_interaction_ids,
            uid,
        )

    def save_to_sql(self, conn: sqlite3.Connection):
        cursor = conn.cursor()
        # Insert a new row into the messages table
        cursor.execute(
            "INSERT INTO interactions (created_at, id, user_message_id, response_message_id) VALUES (?, ?, ?, ?);",
            (self.created_at, self.uid, self.user_message_id, self.response_message_id),
        )

        # Insert the relationships into the related_messages table
        for related_message_id in self.relevant_interaction_ids:
            cursor.execute(
                "INSERT INTO interaction_relevant_interactions (interaction_id, related_interaction_id) VALUES (?, ?);",
                (self.uid, related_message_id),
            )

        # Insert the relationships into the related_messages table
        for related_message_id in self.recent_interaction_ids:
            cursor.execute(
                "INSERT INTO interaction_recent_interactions (interaction_id, related_interaction_id) VALUES (?, ?);",
                (self.uid, related_message_id),
            )

        # Insert the relationships into the related_messages table
        for system_message_id in self.system_message_ids:
            cursor.execute(
                "INSERT INTO interaction_system_messages (interaction_id, system_message_id) VALUES (?, ?);",
                (self.uid, system_message_id),
            )
        conn.commit()

    @classmethod
    def sql_tables(cls):
        return [
            (
                "CREATE TABLE IF NOT EXISTS interactions ("
                "   created_at DATETIME,"
                "   id TEXT PRIMARY KEY,"
                "   user_message_id TEXT NOT NULL,"
                "   response_message_id TEXT NOT NULL,"
                "   FOREIGN KEY (user_message_id) REFERENCES messages (id) ON DELETE CASCADE,"
                "   FOREIGN KEY (response_message_id) REFERENCES messages (id) ON DELETE CASCADE"
                ")"
            ),
            (
                "CREATE TABLE IF NOT EXISTS interaction_system_messages ("
                "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "   interaction_id TEXT NOT NULL,"
                "   system_message_id TEXT NOT NULL,"
                "   FOREIGN KEY (interaction_id) REFERENCES interactions (id) ON DELETE CASCADE,"
                "   FOREIGN KEY (system_message_id) REFERENCES messages (id) ON DELETE CASCADE"
                ")"
            ),
            (
                "CREATE TABLE IF NOT EXISTS interaction_relevant_interactions ("
                "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "    interaction_id TEXT NOT NULL,"
                "    related_interaction_id TEXT NOT NULL,"
                "    FOREIGN KEY (interaction_id) REFERENCES interactions (id) ON DELETE CASCADE,"
                "    FOREIGN KEY (related_interaction_id) REFERENCES interactions (id) ON DELETE CASCADE"
                ")"
            ),
            (
                "CREATE TABLE IF NOT EXISTS interaction_recent_interactions ("
                "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "    interaction_id TEXT NOT NULL,"
                "    related_interaction_id TEXT NOT NULL,"
                "    FOREIGN KEY (interaction_id) REFERENCES interactions (id) ON DELETE CASCADE,"
                "    FOREIGN KEY (related_interaction_id) REFERENCES interactions (id) ON DELETE CASCADE"
                ")"
            ),
        ]
