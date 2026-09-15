"""The container files, parsed: the test suite never runs Docker.

Without a build, these checks are all the verification the Dockerfile,
.dockerignore and docker-compose.yml get. They pin what must not regress:

- nothing under data/, runs/ or models/ (third-party statute text, uploaded
  documents, model weights) is copied into the image, the build context leaves
  those directories out, and no RUN can download a model while the image builds;
- the app runs as a non-root user that owns the directories it writes;
- the port is published on the host's loopback only;
- the image looks for its data and its model cache where compose mounts them;
- the image installs its requirements under constraints.txt, and a CUDA torch
  fails the build;
- compose never creates the data directory on the host, and the container
  runs with an init process, no capabilities and no way to gain privileges.

The Dockerfile and .dockerignore are read as Docker reads them; a harmless
rewrite (`EXPOSE 8000/tcp`, `USER 1000`, `data/**`) still passes. The compose
file is parsed with PyYAML, which requirements-ci.txt pins.
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
# All the image may copy from the build context: the app, the constraints its
# requirements install under, and the synthetic demo document it is shown with
# (evals/app/README.md). Never `.`.
IMAGE_SOURCES = {"src", "ui", "tasks.py", "requirements.txt", "constraints.txt", "evals/app"}
APP_CMD = ["python", "tasks.py", "app", "--host", "0.0.0.0", "--port", "8000"]
# Anything that would fetch a model while the image builds: the weights belong
# in the hf-cache volume at run time, never in a layer.
BUILD_TIME_DOWNLOAD = re.compile(
    r"SentenceTransformer|snapshot_download|from_pretrained|\bhf\s+download\b|huggingface-cli|tasks\.py")
SHELL_SEPARATORS = {"&&", "||", ";", "|", "&"}
# .dockerignore patterns that each exclude a whole top-level directory d.
WHOLE_DIRECTORY = ("{d}", "{d}/**", "{d}/*", "**/{d}", "**/{d}/**")


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _instructions(dockerfile: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) per instruction, as Docker reads the file: a
    trailing backslash continues the line, and comment lines are dropped, even
    inside a continued instruction."""
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


def _commands(run: str) -> list[list[str]]:
    """A RUN's commands, each a list of words: its flags dropped, then split
    on && || ; | and &, with quoted text kept whole."""
    run = re.sub(r"^(?:--\S+\s+)*", "", run)
    if run.startswith("["):
        return [json.loads(run)]  # exec form: one command, no shell
    lexer = shlex.shlex(run, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    commands: list[list[str]] = [[]]
    for token in lexer:
        if token in SHELL_SEPARATORS:
            commands.append([])
        else:
            commands[-1].append(token)
    return [command for command in commands if command]


def _run_commands(instructions: list[tuple[str, str]]) -> list[list[str]]:
    return [command for arguments in _all(instructions, "RUN") for command in _commands(arguments)]


def _copy_sources(arguments: str) -> list[str]:
    """A COPY or ADD's sources from the build context, flags left out; none
    for `--from`, which copies from another stage or image."""
    parts = json.loads(arguments) if arguments.startswith("[") else shlex.split(arguments)
    if any(part.startswith("--from") for part in parts):
        return []
    return [part for part in parts if not part.startswith("--")][:-1]


def _env(instructions: list[tuple[str, str]]) -> dict[str, str]:
    env: dict[str, str] = {}
    for arguments in _all(instructions, "ENV"):
        words = shlex.split(arguments)
        if words and "=" not in words[0]:
            env[words[0]] = " ".join(words[1:])  # the legacy `ENV KEY value` form
        else:
            env.update(word.partition("=")[::2] for word in words)
    return env


def _created_users(instructions: list[tuple[str, str]]) -> dict[str, str | None]:
    """name -> uid (None when unset) for each user a RUN creates."""
    users: dict[str, str | None] = {}
    for command in _run_commands(instructions):
        if command[0] in ("useradd", "adduser"):
            uid = next((command[i + 1] for i, word in enumerate(command[:-1]) if word in ("-u", "--uid")), None)
            users[command[-1]] = uid or next((w.split("=", 1)[1] for w in command if w.startswith("--uid=")), None)
    return users


def _chowned(instructions: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(owner, absolute path) for every path a RUN hands to chown."""
    pairs = []
    for command in _run_commands(instructions):
        if command[0] == "chown":
            words = [word for word in command[1:] if not word.startswith("-")]
            pairs += [(words[0].split(":")[0], path) for path in words[1:] if path.startswith("/")]
    return pairs


def _compose() -> dict:
    yaml = pytest.importorskip("yaml")  # pinned in requirements-ci.txt
    return yaml.safe_load(_read("docker-compose.yml"))


def _mounts(service: dict) -> dict[str, tuple[str, str]]:
    """target -> (type, source) for each of a service's volumes, short or long syntax."""
    mounts = {}
    for volume in service.get("volumes") or []:
        if isinstance(volume, dict):
            mounts[volume["target"]] = (volume.get("type", "volume"), str(volume.get("source", "")))
        else:
            source, target = volume.split(":")[:2]
            mounts[target] = ("bind" if source.startswith((".", "/", "~")) else "volume", source)
    return mounts


def _pairs(entries, separator: str) -> dict[str, str]:
    """A compose list (`KEY=value`, `host:ip`) or mapping, as a dict."""
    if isinstance(entries, dict):
        return {str(key): str(value) for key, value in entries.items()}
    return dict(re.split(f"[{separator}]", str(entry), maxsplit=1) for entry in entries or [])


def test_nothing_under_data_runs_or_models_is_copied_into_the_image():
    instructions = _instructions(_read("Dockerfile"))
    sources = [posixpath.normpath(source) for keyword in ("COPY", "ADD")
               for arguments in _all(instructions, keyword) for source in _copy_sources(arguments)]

    assert sources, "the Dockerfile copies nothing from the build context"
    for source in sources:
        assert source.lstrip("/").split("/")[0] not in DATA_DIRS, f"{source} would put data or weights in the image"
    # An allowlist, so `COPY . .` fails here as well: it copies whatever the context holds.
    assert set(sources) <= IMAGE_SOURCES, sorted(set(sources) - IMAGE_SOURCES)


def test_no_run_can_download_a_model_while_the_image_builds():
    runs = _all(_instructions(_read("Dockerfile")), "RUN")

    assert not [run for run in runs if BUILD_TIME_DOWNLOAD.search(run)]


def test_the_app_runs_as_a_non_root_user_that_owns_what_it_writes():
    stage = _final_stage(_instructions(_read("Dockerfile")))
    users, created = _all(stage, "USER"), _created_users(stage)

    assert users, "no USER instruction: the app would run as root"
    user = users[-1].split(":")[0]
    assert user not in ("root", "0"), f"USER {users[-1]} is root"
    names = [(name, uid) for name, uid in created.items() if user in (name, uid)]
    assert names, f"USER {user} is not a user this Dockerfile creates"
    # A named volume takes the ownership of the directory it is mounted over, so
    # a root-owned cache directory would leave the encoder nowhere to download to.
    env, owners = _env(stage), set(names[0]) - {None}
    for directory in (env["LEGALRAG_DATA_DIR"], env["HF_HOME"]):
        assert any(owner in owners and (directory == path or directory.startswith(path.rstrip("/") + "/"))
                   for owner, path in _chowned(stage)), f"{directory} is not handed to {user}"


def test_the_port_is_published_on_the_hosts_loopback_only():
    app = _compose()["services"]["app"]
    ports = [f"{p.get('host_ip', '')}:{p.get('published', '')}:{p.get('target', '')}" if isinstance(p, dict)
             else str(p) for p in app.get("ports") or []]

    assert "127.0.0.1:8000:8000" in ports, ports
    assert all(port.startswith("127.0.0.1:") for port in ports), ports
    # Any other network mode, the host's above all, would bypass what is published.
    assert "network_mode" not in app


def test_the_build_context_leaves_out_the_data_directories():
    patterns = [line.strip() for line in _read(".dockerignore").splitlines()
                if line.strip() and not line.strip().startswith("#")]
    excluded = {posixpath.normpath(p.lstrip("/")) for p in patterns if not p.startswith("!")}
    let_back_in = [posixpath.normpath(p[1:].lstrip("/")) for p in patterns if p.startswith("!")]

    for directory in DATA_DIRS:
        assert excluded & {form.format(d=directory) for form in WHOLE_DIRECTORY}, f"{directory}/ is not in .dockerignore"
        assert not any(p.split("/")[0] in (directory, "**") for p in let_back_in), \
            f"an exception lets {directory}/ back into the context"


def test_the_image_looks_for_its_data_and_model_cache_where_compose_mounts_them():
    env = _env(_final_stage(_instructions(_read("Dockerfile"))))
    compose = _compose()
    mounts = _mounts(compose["services"]["app"])

    assert mounts.get(env["LEGALRAG_DATA_DIR"]) == ("bind", "./data/app"), "uploads would die with the container"
    assert mounts.get(env["HF_HOME"]) == ("volume", "hf-cache"), "the encoder would download again in every new container"
    assert "hf-cache" in (compose.get("volumes") or {}), "hf-cache is not declared as a named volume"


def test_the_app_listens_on_all_container_interfaces_and_the_healthcheck_never_asks_ollama():
    stage = _final_stage(_instructions(_read("Dockerfile")))
    cmd = _all(stage, "CMD")[-1]

    assert cmd.startswith("["), "CMD in shell form: the app would not receive the stop signal"
    assert json.loads(cmd) == APP_CMD
    assert "8000" in {word.split("/")[0] for arguments in _all(stage, "EXPOSE") for word in arguments.split()}
    # GET /, never /api/health: health asks Ollama twice on every call.
    assert re.search(r"""["']http://(?:127\.0\.0\.1|localhost):8000/?["']""", _all(stage, "HEALTHCHECK")[-1])


def test_the_app_service_is_built_here_and_reaches_ollama_on_the_host():
    app = _compose()["services"]["app"]
    build = app.get("build")

    assert (build.get("context") if isinstance(build, dict) else build) == "."
    environment = _pairs(app.get("environment"), "=")
    assert environment.get("OLLAMA_HOST") == "http://host.docker.internal:11434"
    assert environment.get("LEGALRAG_MODEL") == "ollama:gemma3:4b"
    assert _pairs(app.get("extra_hosts"), ":=").get("host.docker.internal") == "host-gateway"


# --- one pinned set for the image: constraints.txt, and no CUDA torch --------------

CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _pip_installs(stage: list[tuple[str, str]]) -> list[tuple[tuple[int, int], list[str]]]:
    """((instruction index, command index), words) for each pip install a RUN makes, in build order."""
    return [((i, j), command) for i, (keyword, arguments) in enumerate(stage) if keyword == "RUN"
            for j, command in enumerate(_commands(arguments))
            if command[:2] in (["pip", "install"], ["pip3", "install"]) or command[1:4] == ["-m", "pip", "install"]]


def _constraint_files(command: list[str]) -> list[str]:
    return ([command[i + 1] for i, word in enumerate(command[:-1]) if word in ("-c", "--constraint")]
            + [word.split("=", 1)[1] for word in command if word.startswith("--constraint=")])


def _requirements(name: str) -> list[str]:
    lines: list[str] = []
    for raw in _read(name).splitlines():
        line = re.sub(r"(?:^|\s)#.*$", "", raw).strip()
        if line.startswith(("-r ", "--requirement ")):
            lines += _requirements(line.split(maxsplit=1)[1])
        elif line and not line.startswith("-"):
            lines.append(line)
    return lines


def test_the_image_installs_under_constraints_and_refuses_a_cuda_torch():
    stage = _final_stage(_instructions(_read("Dockerfile")))
    installs = _pip_installs(stage)
    from_cpu_index = [command for _, command in installs if any(CPU_INDEX in word for word in command)]
    constrained = [(position, command) for position, command in installs if command not in from_cpu_index]

    assert constrained, "nothing installs the requirements"
    for _, command in constrained:
        # Without the constraints, a release asking for a newer torch swaps in the CUDA build from PyPI.
        assert "constraints.txt" in _constraint_files(command), f"unconstrained: {' '.join(command)}"
    torch_pin = next(line for line in _read("constraints.txt").splitlines() if line.startswith("torch=="))
    assert all(torch_pin in command for command in from_cpu_index), f"torch from the CPU index is not {torch_pin}"
    copied = [i for i, (keyword, arguments) in enumerate(stage)
              if keyword in ("COPY", "ADD") and "constraints.txt" in _copy_sources(arguments)]
    assert copied and copied[0] < constrained[0][0][0], "constraints.txt is not copied before it is used"
    checks = [(i, j) for i, (keyword, arguments) in enumerate(stage) if keyword == "RUN"
              for j, command in enumerate(_commands(arguments))
              if "torch" in " ".join(command) and "version.cuda" in " ".join(command)]
    assert any(check > installs[-1][0] for check in checks), "no torch.version.cuda check after the last install"


def test_the_image_installs_no_test_tooling():
    names = {re.split(r"[\s<>=!~\[;@]", line, maxsplit=1)[0].lower() for line in _requirements("requirements.txt")}

    assert names, "requirements.txt names nothing"
    assert "pytest" not in names, "pytest belongs in requirements-dev.txt, out of the image"


# --- compose on a Linux host: no root-owned data directory, a hardened container ----


def test_compose_never_creates_the_data_directory_on_the_host():
    env = _env(_final_stage(_instructions(_read("Dockerfile"))))
    mounts = [volume for volume in _compose()["services"]["app"].get("volumes") or []
              if isinstance(volume, dict) and volume.get("target") == env["LEGALRAG_DATA_DIR"]]

    # The short syntax, or create_host_path left on, lets Docker create a missing
    # ./data/app owned by root, which the app (uid 1000) cannot write.
    assert mounts, "the data mount is not in the long syntax"
    assert mounts[0].get("type") == "bind"
    assert (mounts[0].get("bind") or {}).get("create_host_path") is False


def test_the_app_container_is_hardened():
    app = _compose()["services"]["app"]
    security = {str(option).replace("=", ":") for option in app.get("security_opt") or []}

    assert app.get("init") is True, "no init process: PID 1 neither forwards signals nor reaps children"
    assert "ALL" in {str(cap).upper() for cap in app.get("cap_drop") or []}, "capabilities are not all dropped"
    assert not app.get("cap_add"), "a capability is added back"
    assert security & {"no-new-privileges", "no-new-privileges:true"}, "a setuid binary could still gain privileges"
    assert not app.get("privileged"), "privileged undoes every restriction"
