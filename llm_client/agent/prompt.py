from typing import Any
import logging

import tiktoken

from llm_client.types.openai import Message
from llm_client.config import Config
from llm_client.agent.memory.sql_backed_memory_objects import SqlMessage
from llm_client.agent.memory.remembered_interaction import RememberedInteraction


cfg = Config()
logger = logging.getLogger()

tokens_per_message = {"gpt-3.5-turbo": 4, "gpt-3.5-turbo-0301": 4, "gpt-4": 3, "gpt-4-0314": 3}

tokens_per_name = {"gpt-3.5-turbo": -1, "gpt-3.5-turbo-0301": -1, "gpt-4": 1, "gpt-4-0314": 1}


def count_message_tokens(message: Message, model: str = "gpt-3.5-turbo-0301") -> int:
    """
    Returns the number of tokens used by a list of messages.

    Args:
        messages (Message): A Message containing role and content.
        model (str): The name of the model to use for tokenization.
            Defaults to "gpt-3.5-turbo-0301".

    Returns:
        int: The number of tokens used by the message.
    """
    if model not in tokens_per_message:
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for model {model}.\n"
            " See https://github.com/openai/openai-python/blob/main/chatml.md for"
            " information on how messages are converted to tokens."
        )

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warn("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return tokens_per_message[model] + len(encoding.encode(message.content)) + len(encoding.encode(message.role))


def count_msg_dict(msg_dict: dict[str, list[Message]]):
    messages = msg_dict["user"] + msg_dict["instruction"] + msg_dict["memory"]
    return count_message_tokens(messages)


class ExceededTokenLimit(Exception):
    pass


class Prompt:
    history_intro: str = "The following "
    token_limit = 1000

    def __init__(self):
        self.user_message: SqlMessage = None
        self.system_messages: list[SqlMessage] = []
        self.recent_interactions: list[RememberedInteraction] = []
        self.relevant_interactions: list[RememberedInteraction] = []
        self._messages: dict[str, list[Message]] = {}
        self._num_tokens = 0

    def add_user_message(self, user_message: SqlMessage):
        next_message = Message(role="user", content=user_message.text)
        token_size = count_message_tokens(next_message)
        new_token_count = self._num_tokens + token_size
        if new_token_count <= self.token_limit:
            self.user_message = user_message
            self._messages.setdefault("user", []).append(next_message)
            self._num_tokens = new_token_count
        else:
            raise ExceededTokenLimit(
                f"The message {user_message.text} has a token length of {token_size}. This exceeds the token limit by {new_token_count - self.token_limit}."
            )

    def add_recent_memory(self, previous_interaction: RememberedInteraction):
        memory_msg = (
            f"MemoryLog-{previous_interaction.created_at}: "
            f'Remember when I said, "{previous_interaction.user_message}" you replied with "{previous_interaction.response_message}".'
        )
        next_message = Message(role="user", content=memory_msg)
        new_token_count = self._num_tokens + count_message_tokens(next_message)
        if new_token_count <= self.token_limit:
            self.recent_interactions.append(previous_interaction)
            self._messages.setdefault("recent_memory", []).append(next_message)
            self._num_tokens = new_token_count
        else:
            raise ExceededTokenLimit("Addition of message would exceed token limit.")

    def add_relevant_memory(self, previous_interaction: RememberedInteraction):
        memory_msg = (
            f"MemoryLog-{previous_interaction.created_at}: "
            f'We talked about something similar previously when I said, "{previous_interaction.user_message}" and you replied with "{previous_interaction.response_message}".'
        )
        next_message = Message(role="user", content=memory_msg)
        new_token_count = self._num_tokens + count_message_tokens(next_message)
        if new_token_count <= self.token_limit:
            self.relevant_interactions.append(previous_interaction)
            self._messages.setdefault("relevant_memory", []).append(next_message)
            self._num_tokens = new_token_count
        else:
            raise ExceededTokenLimit("Addition of message would exceed token limit.")

    def add_system_message(self, system_message: SqlMessage):
        next_message = Message(role="system", content=system_message.text)
        new_token_count = self._num_tokens + count_message_tokens(next_message)
        if new_token_count <= self.token_limit:
            self._messages.setdefault("system", []).append(next_message)
            self._num_tokens = new_token_count
        else:
            raise ExceededTokenLimit("Addition of message would exceed token limit.")

    @property
    def chat(self):
        messages: list[Message] = self._messages.get("system", [])
        messages += self._messages.get("relevant_memory", [])
        messages += self._messages.get("recent_memory", [])
        messages += self._messages.get("user", [])
        return [message.dict() for message in messages if len(message.content) > 0]

    @property
    def token_count(self):
        return len(self)

    def __len__(self):
        return self._num_tokens
