from typing import Any
import os
import openai
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime
import csv

from llm_client.config import Config

from llm_client.abilities.github_search import get_merged_issues_last_day
from llm_client.session.session_base import SessionBase


load_dotenv()
cfg = Config()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)


def save_dict_to_csv(data: dict[str, Any], filename: str):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    repo_url = "https://github.com/Significant-Gravitas/Auto-GPT"
    merged_issues = get_merged_issues_last_day(repo_url)
    print(len(merged_issues))
    breakpoint()

    issue_summaries = []
    if merged_issues:
        # daily_issues = "\n".join([f'- Issue #{issue["number"]}: {issue["title"]}' for issue in merged_issues])
        # daily_issues = json.dumps(merged_issues, indent="  ")
        # print(daily_issues)
        session = SessionBase(llm_model="gpt-3.5-turbo", temperature=0.7)
        session.add_system_prompt("Summarize the given text.")
        session.add_system_prompt("Parse message as a json object.")
        session.add_system_prompt("Don't enumerate the input.")
        session.add_system_prompt("Limit summary to a single paragraph.")
        session.add_system_prompt("Think about how some of the issues fit together.")
        session.add_system_prompt("Include the name of the issue in the issue summary.")
        session.add_system_prompt("Provide summary regarding the overall goals of the changes.")
        session.add_system_prompt("Lump issues with similar objectives together in the summary.")
        for issue in merged_issues:
            session.query = (
                f"{json.dumps(issue, cls=DateTimeEncoder)}\n\nReturn a single paragraph summary of this issue."
            )
            summary = session.execute()
            issue_summaries.append(summary)
            print(summary)
            print()

        session.query = "{}\n\nReturn a single paragraph summary of these issues.".format("\n".join(issue_summaries))
        result = session.execute()
        print(result)
    else:
        print("No merged issues found in the last day.")


if __name__ == "__main__":
    main()
