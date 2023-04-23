from typing import List
import logging

# from agi.memory.conversation_history import ConversationHistory
from agi.types.openai import Message

logger = logging.getLogger()


class ChatExperiment:
    def __init__(self):
        self._system_prompt = []
        # self.memory = ConversationHistory()

    def add_system_prompt(self, system_prompt: str):
        self._system_prompt.append(system_prompt)

    def list_system_prompt(self):
        for idx, prompt in enumerate(self._system_prompt):
            print(f"system-{idx}, {prompt}")

    def generate_prompt(self, user_message: str):
        messages = []
        for prompt in self._system_prompt:
            messages.append(Message(role="system", content=prompt))
        
        from message in self.memory.k:
