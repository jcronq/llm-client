from typing import Any
import os
import openai
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import json
import csv

from llm_client.config import Config

# from llm_client.llm_utils import create_chat_completion, create_embedding_with_ada
# from llm_client.types.openai import Message, Role
# from llm_client.experiments.information_extraction_system_prompt import (
#     baseline,
#     exp_prompt_as_input,
#     exp_system_prompt_as_input,
# )
from llm_client.session.experiment_runner import ExperimentRunner


load_dotenv()
cfg = Config()


def save_dict_to_csv(data: dict[str, Any], filename: str):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simple CLI that takes a string input and prints it to console")
    parser.add_argument("experiment_file", type=str, help="The string input to print to console")

    args = parser.parse_args()

    experiment = Path(args.experiment_file)

    experiment = ExperimentRunner(experiment)
    results = experiment.run_experiment()
    result_stem = f"{Path(args.experiment_file).stem}-results"
    with open(f"{result_stem}.json", "w") as _f:
        _f.write(json.dumps(results))
    save_dict_to_csv(results, f"{result_stem}.csv")


if __name__ == "__main__":
    main()
