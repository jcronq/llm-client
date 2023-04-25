from typing import Optional

from llm_client.types.openai import Message
from llm_client.llm_utils import create_chat_completion


class SessionBase:
    def __init__(self, llm_model: Optional[str] = None, temperature: Optional[float] = 0.7):
        self._system_prompt: list[str] = []
        self.query_template: str = "{}"
        self.query = ""
        self.model = llm_model or "gpt-3.5-turbo"
        self.temperature = temperature

    def add_system_prompt(self, system_prompt: str):
        self._system_prompt.append(system_prompt)

    def set_model(self, model: str):
        self.model = model

    def set_temperature(self, temperature: str):
        self.temperature = temperature

    def list_system_prompt(self):
        for idx, prompt in enumerate(self._system_prompt):
            print(f"system-{idx}: {prompt}")

    def generate_query_prompt(self, query: Optional[str] = None) -> list[Message]:
        if query is None:
            query = self.query
        messages = []
        for prompt in self._system_prompt:
            messages.append(Message(role="system", content=prompt))

        messages.append(Message(role="user", content=query))
        return messages

    def execute(self):
        prompt = self.generate_query_prompt()
        return create_chat_completion(prompt, self.model, self.temperature)

    def __str__(self):
        prompt = "\n".join(f"system: {prompt}" for prompt in self._system_prompt)
        return "\n\n".join(
            [
                prompt,
                f"user: {self.query}",
            ]
        )
