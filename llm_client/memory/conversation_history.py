from typing import TypeAlias

from pydantic import BaseModel
import numpy as np

from llm_client.llm_utils import create_embedding_with_ada
from llm_client.memory.types import MemoryBlock


class ConversationHistory:
    def __init__(self):
        self.memory: list[MemoryBlock] = []

    def add_back_and_forth(self, user_input: str, assistant_output: str):
        self.memory.append(MemoryBlock(user_input, assistant_output))

    @property
    def user_inputs(self) -> list[str]:
        return [memory_block.user_input for memory_block in self.memory]

    @property
    def assistant_outputs(self) -> list[str]:
        return [memory_block.assistant_output for memory_block in self.memory]

    @property
    def conversations(self) -> list[str]:
        return [memory_block.conversation for memory_block in self.memory]

    @property
    def user_input_embeddings(self) -> list[str]:
        return [memory_block.input_embedding for memory_block in self.memory]

    @property
    def assistant_output_embeddings(self) -> list[str]:
        return [memory_block.output_embedding for memory_block in self.memory]

    @property
    def conversation_embeddings(self) -> list[str]:
        return [memory_block.conversation_embedding for memory_block in self.memory]

    def k_most_similar(self, text: str, k: int) -> list[MemoryBlock]:
        """ "
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        embedding = create_embedding_with_ada(text)

        # What if scores were weighteb as well by how often they come up?  What would that look like?
        scores = np.dot(self.data.embeddings, embedding)

        top_k_indices = np.argsort(scores)[-k:][::-1]
        for i in top_k_indices:
            yield self.messages.texts[i]

    def most_similar_inputs(self, text: str, k: int) -> list[MemoryBlock]:
        """ "
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        embedding = create_embedding_with_ada(text)

        # What if scores were weighteb as well by how often they come up?  What would that look like?
        scores = np.dot(self.data.embeddings, embedding)

        top_k_indices = np.argsort(scores)[-k:][::-1]
        for i in top_k_indices:
            yield self.messages.texts[i]

    def k_most_recent(self, k: int):
        for text in self.memory[: -(k + 1), -1]:
            yield text
