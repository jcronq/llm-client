from typing import Any
from datetime import datetime
import json
import csv
from dotenv import load_dotenv

import smtplib
from email.message import EmailMessage

from llm_client.config import Config
from llm_client.abilities.github_search import get_merged_issues_last_day
from llm_client.session.session_base import SessionBase
from llm_client.storage.sqlite import PullRequestDatabase


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


def send_email(recipient_email, subject, body):
    # Create the email message
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = cfg.sender_email
    msg["To"] = recipient_email

    # Connect to the Gmail SMTP server
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        # Log in to your Gmail account
        server.login(cfg.sender_email, cfg.sender_email_password)

        # Send the email
        server.send_message(msg)


def main():
    db = PullRequestDatabase("pull_requests.db")
    repo_url = "https://github.com/Significant-Gravitas/Auto-GPT"
    merged_issues = get_merged_issues_last_day(repo_url)

    issue_summaries = []
    if merged_issues:
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
            summary = db.get_pull_request(issue["number"])
            if summary is None:
                session.query = (
                    f"{json.dumps(issue, cls=DateTimeEncoder)}\n\nReturn a single paragraph summary of this issue."
                )
                summary = session.execute()
                db.add_pull_request(issue["number"], summary, datetime.now())
                issue["summary"] = summary
                print(summary)
            else:
                (issue["summary"],) = summary
                print(summary[0])
            print()

        session.query = "{}\n\nReturn a single paragraph summary of these issues.".format(
            "\n".join([issue["summary"] for issue in merged_issues])
        )
        result = session.execute()
        print(result)

        issues_str = "\n\n".join([f"{issue['title']}:\n{issue['summary']}" for issue in merged_issues])
        body = (
            "Here is a summary of all the changes to Auto-GPT over the last 24 hours.\n"
            "\n"
            f"{result}\n"
            "\n"
            "Issue Summaries\n"
            f"\n"
            f"{issues_str}"
        )
        send_email(cfg.sender_email, "Auto-GPT Change Summary", body)
    else:
        print("No merged issues found in the last day.")


if __name__ == "__main__":
    main()
