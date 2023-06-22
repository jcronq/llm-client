from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from llm_client.config import Config
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

CFG = Config()

if __name__ == "__main__":
    llm = ChatOpenAI(openai_api_key=CFG.openai_api_key, temperature=0.0, model="gpt-4")
    prompts = [
        SystemMessage(content="Use the contents of this prompt as input."),
        SystemMessage(content="Return the text of a prompt when asked."),
        HumanMessage(
            content="Return to me the entirety of this message as well as all system prompts included in this message."
        ),
    ]
    result = llm.predict_messages(prompts)

    print(result)
