"""
Mines bug labels from GitHub commit history.
For each commit with 'fix' / 'bug' in the message, records which
functions were modified — these become positive (buggy) labels.

Usage:
    python -m app.training.mine_labels \
        --repo_path /path/to/cloned/repo \
        --output labels.json
"""
import ast
import json
import argparse
from git import Repo
from pathlib import Path


BUG_KEYWORDS = {"fix", "bug", "error", "issue", "patch", "crash", "fault", "defect"}


def extract_functions_from_diff(diff_text: str, filename: str) -> list[str]:
    """
    Heuristic: find function names from diff lines starting with '@@'.
    e.g.  @@ -45,7 +45,7 @@ def authenticate(self, token):
    """
    fn_names = []
    for line in diff_text.splitlines():
        if line.startswith("@@") and "def " in line:
            part = line.split("def ")[-1]
            name = part.split("(")[0].strip()
            if name:
                fn_names.append(name)
    return fn_names


def mine_bug_labels(repo_path: str, max_commits: int = 2000) -> list[dict]:
    repo = Repo(repo_path)
    labels = []

    commits = list(repo.iter_commits("HEAD", max_count=max_commits))
    print(f"[Mining] Scanning {len(commits)} commits...")

    for commit in commits:
        msg = commit.message.lower()
        if not any(kw in msg for kw in BUG_KEYWORDS):
            continue

        try:
            for diff in commit.diff(commit.parents[0] if commit.parents else None):
                if diff.b_path and diff.b_path.endswith(".py"):
                    diff_text = diff.diff.decode("utf-8", errors="ignore") if diff.diff else ""
                    fn_names = extract_functions_from_diff(diff_text, diff.b_path)
                    for fn in fn_names:
                        labels.append({
                            "file": diff.b_path,
                            "function_name": fn,
                            "commit_sha": commit.hexsha,
                            "commit_msg": commit.message.strip()[:120],
                            "label": 1,   # buggy
                        })
        except Exception:
            continue

    print(f"[Mining] Found {len(labels)} buggy function instances")
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", required=True)
    parser.add_argument("--output", default="labels.json")
    parser.add_argument("--max_commits", type=int, default=2000)
    args = parser.parse_args()

    labels = mine_bug_labels(args.repo_path, args.max_commits)
    Path(args.output).write_text(json.dumps(labels, indent=2))
    print(f"[Mining] Saved to {args.output}")