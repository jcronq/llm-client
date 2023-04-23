import os
import openai
import subprocess
from dotenv import load_dotenv

from llm_client.config import Config
from llm_client.llm_utils import create_chat_completion, create_embedding_with_ada
from llm_client.types.openai import Message, Role


load_dotenv()
cfg = Config()


def main():
    response = create_embedding_with_ada("Hello World!")
    # response = create_chat_completion(
    #     messages=[Message(role=Role.System, content="this is a test.  Test 1, 2, ")],
    #     model=cfg.fast_llm_model,
    #     temperature=cfg.temperature,
    #     max_tokens=None,
    # )
    breakpoint()


if __name__ == "__main__":
    main()
