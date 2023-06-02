import json
from pathlib import Path

import logging
from llm_client.agent.memory.memory import Memory
from llm_client.agent.prompt import Prompt, ExceededTokenLimit
from llm_client.llm_utils import create_chat_completion
from llm_client.types.openai import Role
from llm_client.logs import logger, print_assistant_thoughts


class Agent:
    default_system_prompts = [
        "Think carefully about your responses.",
        "Work towards the enumerated system objectives.",
        "Use system messages that start with MemoryLog- as input.",
        "Use MemoryLog- messages to understand the context of the current message.",
        "MemoryLog- messages are previous messages in our current conversation.",
        "You have access to previous conversations by reading the MemoryLog- messages",
    ]
    savefile_path = Path("agent-config.json")

    def __init__(self, model: str, temperature, db_file: str):
        self.memory = Memory(db_file)
        self.model = model
        self.temperature = temperature

        self._name = ""
        self._description = ""
        self._objectives = []
        self.alive = False

    def load(self) -> bool:
        if not self.savefile_path.exists():
            return False

        try:
            with self.savefile_path.open("r") as _f:
                agent_data = json.loads(_f.read())
            self._name = agent_data["name"]
            self._description = agent_data["description"]
            self._objectives = agent_data["objectives"]
            self.model = agent_data["model"]
            self.temperature = agent_data["temperature"]
        except:
            return False
        return True

    def save(self):
        self.savefile_path.parent.mkdir(exist_ok=True)
        with self.savefile_path.open("w") as _f:
            _f.write(
                json.dumps(
                    {
                        "name": self._name,
                        "description": self._description,
                        "objectives": self._objectives,
                        "model": self.model,
                        "temperature": self.temperature,
                    }
                )
            )

    def agent_configure(self):
        raise NotImplemented()

    def user_configure(self):
        finished_input = "n"
        while finished_input.lower() != "y" and finished_input.lower() != "yes":
            self._name = input("Agent Name:")
            self._description = input("Agent Description:")
            print("Enter objectives for the Agent. Enter empty line when complete.")
            self._objectives = []
            agent_input = "start"
            agent_input = input(f"..{len(self._objectives)+1}:")
            while agent_input != "":
                self._objectives.append(agent_input)
                agent_input = input(f"..{len(self._objectives)+1}:")
            print()
            print(self)
            print()
            finished_input = input("Accept these changes? (y/n)")
        self.save()

    def run_user_loop(self):
        self.alive = True
        while self.alive:
            prompt = Prompt()
            # System Messaging.
            # Build General Instructions.
            for system_prompt in self.default_system_prompts:
                prompt.add_system_message(self.memory.get_message(Role.System, system_prompt))

            # Objective orientation.
            # Enumerate the objectives.
            for idx, objective in enumerate(self._objectives, start=1):
                prompt.add_system_message(self.memory.get_message(Role.System, f"Objective-{idx}: {objective}"))

            input_required = True
            while input_required:
                try:
                    # The current ask.
                    user_msg = input("Message for the Agent:")
                    prompt.add_user_message(self.memory.get_message(Role.User, user_msg))
                    input_required = False
                except ExceededTokenLimit as err:
                    print(err)
                    print()
                    print("Please try a shorter message.")

            # Memory.
            # Pad the remaining token-space with relevant memory.
            try:
                for sql_interaction in self.memory.k_most_similar_inputs(prompt.user_message.text, 5):
                    remembered_interaction = self.memory.render_prior_interaction(sql_interaction)
                    prompt.add_relevant_memory(remembered_interaction)

                for sql_interaction in self.memory.k_most_recent(5):
                    remembered_interaction = self.memory.render_prior_interaction(sql_interaction)
                    prompt.add_recent_memory(remembered_interaction)

            except ExceededTokenLimit:
                print("Message full.")

            logger.debug(json.dumps(prompt.chat))
            response = create_chat_completion(prompt.chat, model=self.model, temperature=self.temperature)
            self.memory.add_interaction(prompt, response)
            print()
            logger.typewriter_log(response)
            print()

    def __str__(self):
        return f"{self._name}\n{self._description}\n{self._objectives}"
