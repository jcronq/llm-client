"""
Template for experimenting with the efficacy of various System Prompts.
"""
from typing import Optional
from pathlib import Path
import logging
import yaml

import numpy as np

from llm_client.types.openai import Message
from llm_client.llm_utils import create_chat_completion

logger = logging.getLogger()


class ExperimentRunner:
    def __init__(self, experiment_config_file: Path):
        with experiment_config_file.open("r") as _f:
            data = yaml.safe_load(_f)
        self._system_prompt = data.get("system_prompts", [])
        self.query_template: str = data.get("query_template", "{}")
        self.query_filler: str = data.get("query_filler", "")
        self.model = data.get("model", "gpt-3.5-turbo")
        self.temperature = data.get("temperature", 0.0)
        self.perturbations = data.get("perturbations", [])

        self.expected_result = data.get("expected_result", None)
        self.default_label = data.get("default_label", "")

    def add_system_prompt(self, system_prompt: str):
        self._system_prompt.append(system_prompt)

    def set_model(self, model: str):
        self.model = model

    def set_temperature(self, temperature: str):
        self.temperature = temperature

    def list_system_prompt(self):
        for idx, prompt in enumerate(self._system_prompt):
            print(f"system-{idx}, {prompt}")

    def generate_query_prompt(self, query: Optional[str] = None) -> list[Message]:
        if query is None:
            query = self.query
        messages = []
        for prompt in self._system_prompt:
            messages.append(Message(role="system", content=prompt))

        messages.append(Message(role="user", content=query))
        return messages

    def calculate_experiment_similarity(self):
        for perturbation in self.perturbations:
            pass
            # TODO: Pick up here

    def run_experiment(self):
        results = []
        for perturbation in self.perturbations:
            query = self.perturb_experiment(perturbation)
            if query is not None:
                print("-----------------------")
                print(self.to_str(query))
                prompt = self.generate_query_prompt(query)
                result = create_chat_completion(prompt, self.model, self.temperature)
                print(f"assistant: {result}")
                print("-----------------------")
                results.append(self.experiment_state(result))
        return results

    def perturb_experiment(self, perturbation) -> bool:
        """Returns true when the perturbation means it should also run an experiment this turn"""
        result = None
        if "query_filler" in perturbation:
            self.query_filler = perturbation["query_filler"]
            result = self.query
        if "query" in perturbation:
            result = perturbation["query"]
        if "query_template" in perturbation:
            self.query_template = perturbation["query_template"]
        if "default_label" in perturbation:
            self.default_label = perturbation["default_label"]
        if "temperature" in perturbation:
            self.set_temperature(perturbation["temperature"])
        if "model" in perturbation:
            self.set_temperature(perturbation["model"])
        if "add_system_prompt" in perturbation:
            self.add_system_prompt(perturbation["add_system_prompt"])
        return result

    def experiment_state(self, result: str):
        return {
            "system_prompts": ",".join(self._system_prompt),
            "query": self.query,
            "temperature": self.temperature,
            "model": self.model,
            "response": result,
            "expected": self.expected_result,
            "labels": self.default_label,
        }

    @property
    def query(self):
        return self.query_template.format(self.query_filler)

    def to_str(self, query: Optional[str] = None):
        prompt = "\n".join(f"system: {prompt}" for prompt in self._system_prompt)
        return "\n\n".join(
            [
                prompt,
                f"user: {query}",
            ]
        )

    def __str__(self):
        prompt = "\n".join(f"system: {prompt}" for prompt in self._system_prompt)
        return "\n\n".join(
            [
                prompt,
                f"user: {self.query}",
            ]
        )
