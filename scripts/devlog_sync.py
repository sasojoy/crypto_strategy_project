#!/usr/bin/env python3
"""Synchronize development log entries to GitHub Issues."""

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import requests

LOG_PATH = Path("DEVELOPMENT_LOG.md")
PROJECT_NAME = "Product Board"


def get_repo() -> Dict[str, str]:
    """Return repository owner and name from git remote."""
    url = (
        subprocess.check_output([
            "git",
            "config",
            "--get",
            "remote.origin.url",
        ], text=True)
        .strip()
    )
    match = re.search(r"[:/]([\w.-]+)/([\w.-]+)\.git$", url)
    if not match:
        raise RuntimeError(f"Unable to parse repository from URL: {url}")
    return {"owner": match.group(1), "repo": match.group(2)}


def parse_entry(index: int, line: str, desc_line: str) -> Optional[Dict]:
    """Parse a single task line and its description."""
    if "(#" in line:
        return None  # already has an issue reference

    content = line[5:].strip()  # remove '- [ ]'

    prefix = ""
    if content.startswith("["):
        m = re.match(r"\[(.*?)\]\s*(.*)", content)
        if m:
            prefix = m.group(1)
            content = m.group(2)

    assignee = None
    m_owner = re.search(r"@owner:([\w-]+)", content)
    if m_owner:
        assignee = m_owner.group(1)
        content = re.sub(r"@owner:[\w-]+", "", content)

    labels: List[str] = []
    for m in re.finditer(r"#label:([\w-]+)", content):
        labels.append(m.group(1))
    content = re.sub(r"#label:[\w-]+", "", content)

    due = None
    m_due = re.search(r"\(due:(\d{4}-\d{2}-\d{2})\)", content)
    if m_due:
        due = m_due.group(1)
        content = re.sub(r"\(due:\d{4}-\d{2}-\d{2}\)", "", content)

    title = content.strip()
    if prefix:
        title = f"[{prefix}] {title}"

    body = ""
    if desc_line.strip().startswith("desc:"):
        body = desc_line.strip()[5:].strip()

    return {
        "index": index,
        "title": title,
        "assignee": assignee,
        "labels": labels,
        "due": due,
        "body": body,
    }


def find_milestone(token: str, repo: Dict[str, str], due: str) -> Optional[int]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo['owner']}/{repo['repo']}/milestones?state=open"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    for ms in resp.json():
        if ms.get("due_on", "").startswith(due):
            return ms["number"]
    return None


def create_issue(token: str, repo: Dict[str, str], task: Dict, milestone: Optional[int]) -> Dict:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    payload = {
        "title": task["title"],
        "body": task["body"],
    }
    if task["assignee"]:
        payload["assignees"] = [task["assignee"]]
    if task["labels"]:
        payload["labels"] = task["labels"]
    if milestone is not None:
        payload["milestone"] = milestone
    url = f"https://api.github.com/repos/{repo['owner']}/{repo['repo']}/issues"
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_or_create_project(token: str, repo: Dict[str, str]) -> str:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    query = (
        """
        query($owner:String!, $name:String!) {
          repository(owner:$owner, name:$name) {
            owner { id }
            projectsV2(first:20) { nodes { id title } }
          }
        }
        """
    )
    variables = {"owner": repo["owner"], "name": repo["repo"]}
    resp = requests.post(
        "https://api.github.com/graphql", json={"query": query, "variables": variables}, headers=headers, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()["data"]["repository"]
    owner_id = data["owner"]["id"]
    for node in data["projectsV2"]["nodes"]:
        if node["title"] == PROJECT_NAME:
            return node["id"]
    mutation = (
        """
        mutation($ownerId:ID!, $title:String!) {
          createProjectV2(input:{ownerId:$ownerId, title:$title}) {
            projectV2 { id }
          }
        }
        """
    )
    variables = {"ownerId": owner_id, "title": PROJECT_NAME}
    resp = requests.post(
        "https://api.github.com/graphql", json={"query": mutation, "variables": variables}, headers=headers, timeout=30
    )
    resp.raise_for_status()
    return resp.json()["data"]["createProjectV2"]["projectV2"]["id"]


def add_to_project(token: str, project_id: str, issue_node_id: str) -> None:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    mutation = (
        """
        mutation($project:ID!, $content:ID!) {
          addProjectV2ItemById(input:{projectId:$project, contentId:$content}) {
            item { id }
          }
        }
        """
    )
    variables = {"project": project_id, "content": issue_node_id}
    resp = requests.post(
        "https://api.github.com/graphql", json={"query": mutation, "variables": variables}, headers=headers, timeout=30
    )
    resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync development log to GitHub issues")
    parser.add_argument("--dry-run", action="store_true", help="do not create issues, just print actions")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print("DEVELOPMENT_LOG.md not found; nothing to do.")
        return

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    tasks = []
    for idx, line in enumerate(lines):
        if line.startswith("- [ ]"):
            desc_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            task = parse_entry(idx, line, desc_line)
            if task:
                tasks.append(task)

    if not tasks:
        print("No new tasks found.")
        return

    token = os.getenv("GITHUB_TOKEN")
    repo = get_repo()

    project_id: Optional[str] = None
    updated = False
    for task in tasks:
        if args.dry_run:
            print(f"Would create issue: {task['title']}")
            continue
        if not token:
            raise RuntimeError("GITHUB_TOKEN is required for syncing")
        milestone = find_milestone(token, repo, task["due"]) if task["due"] else None
        issue = create_issue(token, repo, task, milestone)
        if project_id is None:
            project_id = get_or_create_project(token, repo)
        add_to_project(token, project_id, issue["node_id"])
        lines[task["index"]] = lines[task["index"]].rstrip() + f" (#{issue['number']})\n"
        updated = True

    if updated and not args.dry_run:
        LOG_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
