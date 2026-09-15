"""The CI files, read: CI itself only runs once the repository is published.

- every action a workflow uses is pinned to a full commit SHA, with the release
  it is as a comment, since a tag can be moved to other code;
- Dependabot proposes action updates, so the pins do not go stale silently;
- CI installs its requirements under constraints.txt, like the image;
- a workflow's token can only read the repository, and checkout does not leave
  it in .git/config for the steps after it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
USES_LINE = re.compile(r"""^\s*(?:-\s*)?uses:\s*["']?([^\s"'#]+)["']?\s*(?:#\s*(\S+))?\s*$""", re.MULTILINE)
PINNED = re.compile(r"[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}")


def _workflows() -> dict[str, str]:
    workflows = ROOT / ".github" / "workflows"
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(workflows.glob("*.y*ml"))}


def test_every_action_is_pinned_to_a_full_commit_sha_with_its_release_as_a_comment():
    workflows = _workflows()
    assert workflows, "no workflow found"

    for name, text in workflows.items():
        # Every `uses:` in the file, however written, must be a line this check reads.
        assert len(USES_LINE.findall(text)) == len(re.findall(r"\buses\s*:", text)), f"{name}: a uses: this check cannot read"
        for ref, release in USES_LINE.findall(text):
            if ref.startswith("./"):
                continue  # an action in this repository: nothing to pin
            assert PINNED.fullmatch(ref), f"{name}: {ref} is not pinned to a 40-hex commit SHA"
            assert re.fullmatch(r"v\d+(?:\.\d+){0,2}", release or ""), f"{name}: {ref} has no `# vX.Y.Z` comment"


def test_dependabot_proposes_action_updates_every_week():
    yaml = pytest.importorskip("yaml")  # pinned in requirements-ci.txt, so CI runs this
    config = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))

    updates = [update for update in config["updates"] if update["package-ecosystem"] == "github-actions"]
    assert config["version"] == 2
    assert [(update["directory"], update["schedule"]["interval"]) for update in updates] == [("/", "weekly")]


def test_ci_installs_its_requirements_under_the_constraints():
    installs = re.findall(r"pip install[^\n]*", _workflows()["test.yml"])

    assert installs, "the workflow installs nothing"
    for command in installs:
        assert re.search(r"(?:^|\s)(?:-c|--constraint)[ =]constraints\.txt(?:\s|$)", command), command
        assert "-r requirements-ci.txt" in command, command


def test_a_workflow_can_only_read_the_repository_and_checkout_keeps_no_token():
    yaml = pytest.importorskip("yaml")
    workflows = _workflows()
    assert workflows, "no workflow found"

    for name, text in workflows.items():
        workflow = yaml.safe_load(text)
        assert workflow.get("permissions") == {"contents": "read"}, f"{name}: the token is not limited to contents: read"
        for job_name, job in workflow["jobs"].items():
            # A job's own permissions replace the workflow's for that job.
            granted = job.get("permissions", {})
            assert isinstance(granted, dict) and set(granted.values()) <= {"read", "none"}, \
                f"{name}: job {job_name} widens the token"
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).startswith("actions/checkout@"):
                    persist = (step.get("with") or {}).get("persist-credentials")
                    # Otherwise the token stays in .git/config, readable by every later step.
                    assert str(persist).lower() == "false", f"{name}: checkout keeps the token in .git/config"
