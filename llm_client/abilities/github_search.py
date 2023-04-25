import requests
from datetime import datetime, timedelta

from github import Github
from github.PullRequest import PullRequest

from llm_client.config import Config

cfg = Config()


def get_merged_issues_last_day(repo_url):
    # Extract the repository owner and name from the URL
    repo_parts = repo_url.split("/")
    repo_owner, repo_name = repo_parts[-2], repo_parts[-1]

    # Create a Github instance with your access token (if provided)
    g = Github(cfg.github_api_key)

    # Get the repository object
    repo = g.get_repo(f"{repo_owner}/{repo_name}")

    # Calculate the timestamp for 24 hours ago
    one_day_ago = datetime.utcnow() - timedelta(days=1)

    # Get the pull requests merged in the last day
    repo.get_pulls()
    closed_pull_requests = repo.get_pulls(state="closed", sort="updated", direction="desc")
    merged_pull_requests = []

    for pr in closed_pull_requests:
        if pr.merged_at and pr.merged_at > one_day_ago:
            merged_pull_requests.append(pr)
        elif pr.updated_at < one_day_ago:
            break

    prs = []
    pr: PullRequest
    for pr in merged_pull_requests:
        prs.append(
            {
                "number": pr.number,
                "title": pr.title,
                "author": pr.user.login,
                "description": pr.body,
                "merged_at": pr.merged_at,
                "url": pr.html_url,
            }
        )

    return prs
