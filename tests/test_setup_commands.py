"""`python tasks.py setup` and `make setup`, as a fresh clone runs them.

requirements.txt is what the image installs, so pytest is not in it. Setup
installs requirements-dev.txt instead, the app's requirements plus pytest, under
constraints.txt: the versions the image and CI install. tasks.py is the runner on
Windows and Makefile.txt its alias on Linux, so both run the same install.
Neither runs here: tasks.py's command is recorded, the Makefile's read.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path

from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def _tasks_setup_command() -> list[str]:
    """What `python tasks.py setup` would run, recorded instead of run."""
    spec = importlib.util.spec_from_file_location("tasks_setup_under_test", ROOT / "tasks.py")
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    commands: list[tuple[str, ...]] = []
    tasks._subprocess = lambda *command: commands.append(command) or 0
    tasks.TASKS["setup"]([])
    assert len(commands) == 1, commands
    return list(commands[0])


def _recipe(target: str) -> list[str]:
    """A Makefile.txt target's recipe lines, as written."""
    lines = (ROOT / "Makefile.txt").read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if re.match(rf"{re.escape(target)}\s*:(?!=)", line)]
    assert len(starts) == 1, f"make {target} is defined {len(starts)} times"
    recipe = []
    for line in lines[starts[0] + 1:]:
        if not line.strip() or not line[0].isspace():
            break
        recipe.append(line)
    return recipe


def _values(command: list[str], *options: str) -> list[str]:
    """The values `command` gives any of `options`, as `-c x`, `--constraint x` or `--constraint=x`."""
    return ([command[i + 1] for i, word in enumerate(command[:-1]) if word in options]
            + [word.split("=", 1)[1] for word in command if "=" in word and word.split("=", 1)[0] in options])


def _names(requirements_file: str) -> set[str]:
    """The distributions a requirements file names, `-r` includes followed."""
    names: set[str] = set()
    for raw in (ROOT / requirements_file).read_text(encoding="utf-8").splitlines():
        line = re.sub(r"(?:^|\s)#.*$", "", raw).strip()
        include = re.match(r"(?:-r|--requirement)[\s=]+(\S+)$", line)
        if include:
            names |= _names(include.group(1))
        elif line and not line.startswith("-"):
            names.add(canonicalize_name(re.split(r"[\s<>=!~\[;@]", line, maxsplit=1)[0]))
    return names


def test_tasks_py_setup_installs_the_test_requirements_under_the_constraints():
    command = _tasks_setup_command()

    assert command[:4] == [sys.executable, "-m", "pip", "install"], command
    # The versions the image and CI install, not whatever is newest on PyPI that day.
    assert _values(command, "-c", "--constraint") == ["constraints.txt"], command
    (requirements,) = _values(command, "-r", "--requirement")
    names = _names(requirements)
    assert "pytest" in names, f"{requirements} has no pytest: `python tasks.py test` would fail after setup"
    assert _names("requirements.txt") <= names, f"{requirements} leaves out some of the app's requirements"


def test_make_setup_runs_the_same_install_as_tasks_py_setup():
    recipe = _recipe("setup")

    assert recipe, "make setup has no recipe"
    assert all(line.startswith("\t") for line in recipe), "make needs a tab at the start of each recipe line"
    assert [shlex.split(line) for line in recipe] == [["$(PY)", *_tasks_setup_command()[1:]]]
