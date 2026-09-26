"""`legalrag.provenance` — the commit and worktree state every result file carries."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalrag.provenance import commit_state, worktree_clean  # noqa: E402


def _repo(tmp_path: Path) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (tmp_path / "evals" / "registry").mkdir(parents=True)
    (tmp_path / "evals" / "registry" / "result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return tmp_path


def test_a_committed_tree_is_clean_and_names_its_head(tmp_path):
    repo = _repo(tmp_path)

    state = commit_state(repo)

    assert len(state["source_commit_sha"]) == 40
    assert state["worktree_clean"] is True


def test_an_edited_registry_file_does_not_make_the_tree_dirty(tmp_path):
    repo = _repo(tmp_path)
    (repo / "evals" / "registry" / "result.json").write_text('{"n": 1}', encoding="utf-8")

    assert worktree_clean(repo) is True


def test_an_edit_outside_the_registry_makes_the_tree_dirty(tmp_path):
    repo = _repo(tmp_path)
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")

    assert worktree_clean(repo) is False


def test_outside_a_git_checkout_both_fields_are_none(tmp_path):
    assert commit_state(tmp_path) == {"source_commit_sha": None, "worktree_clean": None}
