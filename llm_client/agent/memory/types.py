from typing import TypeAlias

from pydantic import BaseModel

from llm_client.llm_utils import create_embedding_with_ada

Embedding = list[float]


class MemoryBlock(BaseModel):
    user_input: str
    assistant_output: str
    conversation: str
    input_embedding: Embedding
    output_embedding: Embedding
    conversation_embedding: Embedding

    def __init__(self, user_input, assistant_output):
        conversation = f"user: {user_input}\n\nassistant: {assistant_output}"
        input_embedding = create_embedding_with_ada(user_input)
        output_embedding = create_embedding_with_ada(assistant_output)
        conversation_embedding = create_embedding_with_ada(conversation_embedding)
        super().__init__(
            user_input=user_input,
            assistant_output=assistant_output,
            conversation=conversation,
            input_embedding=input_embedding,
            output_embedding=output_embedding,
            conversation_embedding=conversation_embedding,
        )

    def __repr__(self):
        return f"MemoryBlock({self.user_input}, {self.assistant_output})"

    def __str__(self):
        self.conversation
