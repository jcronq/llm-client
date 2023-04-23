"""
Template for experimenting with the efficacy of various System Prompts.
"""
from typing import List
import logging

from llm_client.memory.conversation_history import ConversationHistory
from llm_client.types.openai import Message

logger = logging.getLogger()


class SystemPromptExperiment:
    def __init__(self):
        self._system_prompt = []
        self._test_query
        self.memory = ConversationHistory()

    def add_system_prompt(self, system_prompt: str):
        self._system_prompt.append(system_prompt)

    def set_test_query(self, system_prompt: str):
        self._system_prompt.append(system_prompt)

    def list_system_prompt(self):
        for idx, prompt in enumerate(self._system_prompt):
            print(f"system-{idx}, {prompt}")

    def generate_query(self, user_message: str):
        messages = []
        for prompt in self._system_prompt:
            messages.append(Message(role="system", content=prompt))

        messages.append(Message(role="user", content=self._test_query))

    def __str__(self):
        prompt = "\n".join(f"system: {prompt}" for prompt in self._system_prompt)
        return "\n\n".join(
            [
                prompt,
                self._test_query,
            ]
        )
