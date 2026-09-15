"""The container files, read as text: the test suite never runs Docker.

Without a build, these checks are all the verification the Dockerfile,
.dockerignore and docker-compose.yml get. They pin what must not regress:

- nothing under data/, runs/ or models/ (third-party statute text, uploaded
  documents, model weights) is copied into the image, and the build context
  leaves those directories out;
- the app runs as a non-root user that owns the directories it writes;
- the port is published on the host's loopback only;
- the image looks for its data and its model cache where compose mounts them.

The compose file is read line by line, so no YAML parser is needed. Where
PyYAML is installed, one more test parses it properly; CI does not install it.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = ("data", "runs", "models")
# All the image may copy from the build context: the app, and the synthetic
# demo document it is shown with (evals/app/README.md). Never `.`.
IMAGE_SOURCES = {"src", "ui", "tasks.py", "requirements.txt", "evals/app"}
APP_CMD = ["python", "tasks.py", "app", "--host", "0.0.0.0", "--port", "8000"]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _instructions(dockerfile: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) per instruction, continuation lines joined and
    comments dropped, as Docker reads them."""
    instructions: list[tuple[str, str]] = []
    pending = ""
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        keyword, _, arguments = (pending + stripped).partition(" ")
        pending = ""
        instructions.append((keyword.upper(), arguments.strip()))
    return instructions


def _final_stage(instructions: list[tuple[str, str]]) -> list[tuple[str, str]]:
    starts = [i for i, (keyword, _) in enumerate(instructions) if keyword == "FROM"]
    return instructions[starts[-1]:] if starts else instructions


def _all(instructions: list[tuple[str, str]], keyword: str) -> list[str]:
    return [arguments for name, arguments in instructions if name == keyword]


def _copy_sources(arguments: str) -> list[str]:
    """A COPY or ADD's sources: every path but the destination, flags left out."""
    parts = json.loads(arguments) if arguments.startswith("[") else shlex.split(arguments)
    return [part for part in parts if not part.startswith("--")][:-1]


def _env(instructions: list[tuple[str, str]]) -> dict[str, str]:
    return dict(pair.partition("=")[::2] for arguments in _all(instructions, "ENV")
                for pair in shlex.split(arguments))


def _chowned(instructions: list[tuple[str, str]]) -> list[str]:
    """Every absolute path a RUN hands to chown."""
    return [word for command in _all(instructions, "RUN") for segment in re.split(r"&&|\|\||;", command)
            if shlex.split(segment)[:1] == ["chown"]
            for word in shlex.split(segment)[1:] if word.startswith("/")]


def _list_items(yaml_text: str, key: str) -> list[str]:
    """The `- item` entries under every `key:` line of a YAML file, unquoted,
    trailing comments removed. Block style only, as docker-compose.yml is written."""
    items: list[str] = []
    key_indent: int | None = None
    for line in yaml_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if key_indent is not None and indent >= key_indent and stripped.startswith("- "):
            items.append(re.sub(r"\s+#.*$", "", stripped[2:]).strip().strip("\"'"))
            continue
        if key_indent is not None and indent > key_indent:
            continue  # a nested mapping under the key, not a list item
        key_indent = indent if stripped == f"{key}:" else None
    return items


def test_nothing_under_data_runs_or_models_is_copied_into_the_image():
    instructions = _instructions(_read("Dockerfile"))
    sources = [posixpath.normpath(source) for keyword in ("COPY", "ADD")
               for arguments in _all(instructions, keyword) for source in _copy_sources(arguments)]

    assert sources, "the Dockerfile copies nothing from the build context"
    for source in sources:
        assert source.lstrip("/").split("/")[0] not in DATA_DIRS, f"{source} would put data or weights in the image"
    # An allowlist, so `COPY . .` fails here as well: it copies whatever the context holds.
    assert set(sources) <= IMAGE_SOURCES, sorted(set(sources) - IMAGE_SOURCES)


def test_the_app_runs_as_a_non_root_user_that_owns_what_it_writes():
    stage = _final_stage(_instructions(_read("Dockerfile")))
    users = _all(stage, "USER")

    assert users, "no USER instruction: the app would run as root"
    assert users[-1].split(":")[0] == "app"
    assert any(re.search(r"\buseradd\b[^&;|]*\bapp\b", command) for command in _all(stage, "RUN")), \
        "the app user is never created"
    # A named volume takes the ownership of the directory it is mounted over, so
    # a root-owned cache directory would leave the encoder nowhere to download to.
    env, chowned = _env(stage), _chowned(stage)
    for directory in (env["LEGALRAG_DATA_DIR"], env["HF_HOME"]):
        assert any(directory == path or directory.startswith(path.rstrip("/") + "/") for path in chowned), \
            f"{directory} is not handed to the app user"


def test_the_port_is_published_on_the_hosts_loopback_only():
    compose = _read("docker-compose.yml")
    ports = _list_items(compose, "ports")

    assert "127.0.0.1:8000:8000" in ports
    assert all(port.startswith("127.0.0.1:") for port in ports), ports
    # Any other network mode, the host's above all, would bypass what is published.
    assert not re.search(r"^\s*network_mode\s*:", compose, re.MULTILINE)


def test_the_build_context_leaves_out_the_data_directories():
    patterns = [line.strip() for line in _read(".dockerignore").splitlines()
                if line.strip() and not line.strip().startswith("#")]
    excluded = {posixpath.normpath(p.lstrip("/")) for p in patterns if not p.startswith("!")}
    let_back_in = [posixpath.normpath(p[1:].lstrip("/")) for p in patterns if p.startswith("!")]

    for directory in DATA_DIRS:
        assert directory in excluded, f"{directory}/ is not in .dockerignore"
        assert not any(p.split("/")[0] in (directory, "**") for p in let_back_in), \
            f"an exception lets {directory}/ back into the context"


def test_the_image_looks_for_its_data_and_model_cache_where_compose_mounts_them():
    env = _env(_final_stage(_instructions(_read("Dockerfile"))))
    compose = _read("docker-compose.yml")
    source_of = {}
    for volume in _list_items(compose, "volumes"):
        source, target = volume.split(":")[:2]
        source_of[target] = source

    assert source_of.get(env["LEGALRAG_DATA_DIR"]) == "./data/app", "uploads would die with the container"
    assert source_of.get(env["HF_HOME"]) == "hf-cache", "the encoder would be downloaded again on every new container"
    assert re.search(r"^\s+hf-cache:\s*$", compose, re.MULTILINE), "hf-cache is not declared as a named volume"


def test_the_app_listens_on_all_container_interfaces_and_the_healthcheck_never_asks_ollama():
    stage = _final_stage(_instructions(_read("Dockerfile")))

    assert json.loads(_all(stage, "CMD")[-1]) == APP_CMD
    assert "8000" in _all(stage, "EXPOSE")
    healthcheck = _all(stage, "HEALTHCHECK")[-1]
    # GET /, never /api/health: health asks Ollama twice on every call.
    assert re.search(r"""["']http://127\.0\.0\.1:8000/["']""", healthcheck), healthcheck


def test_the_compose_file_parses_as_yaml_to_what_the_text_checks_read():
    yaml = pytest.importorskip("yaml")
    text = _read("docker-compose.yml")
    compose = yaml.safe_load(text)
    app = compose["services"]["app"]

    assert list(compose["services"]) == ["app"]
    assert app["build"] == "."
    assert app["ports"] == _list_items(text, "ports") == ["127.0.0.1:8000:8000"]
    assert app["volumes"] == _list_items(text, "volumes") == [
        "./data/app:/data/app", "hf-cache:/home/app/.cache/huggingface"]
    assert sorted(app["environment"]) == [
        "LEGALRAG_MODEL=ollama:gemma3:4b", "OLLAMA_HOST=http://host.docker.internal:11434"]
    assert app["extra_hosts"] == ["host.docker.internal:host-gateway"]
    assert "network_mode" not in app
    assert "hf-cache" in compose["volumes"]
