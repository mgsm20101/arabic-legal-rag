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
    r"SentenceTransformer|snapshot_download|from_pretrained|\bhf\s+download\b|huggingface-cli|tasks\.py"
    r"|load_model|huggingface\.co")
SHELL_SEPARATORS = {"&&", "||", ";", "|", "&"}
# .dockerignore patterns that each exclude a whole top-level directory d.
WHOLE_DIRECTORY = ("{d}", "{d}/**", "{d}/*", "**/{d}", "**/{d}/**")
# A heredoc opens a RUN's body across lines this line-based scanner never
# reconstructs, so it could hide anything from every check below.
HEREDOC = re.compile(r"<<-?\s*['\"]?\w")
# This Dockerfile copies from no external image by stage name or digest; a
# `--from=` may only reference one of its own, earlier stages.
FROM_ALLOWLIST: frozenset[str] = frozenset()


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _instructions(dockerfile: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) per instruction, as Docker reads the file: a
    trailing backslash continues the line, and comment lines are dropped, even
    inside a continued instruction. Splits the keyword from its arguments on
    any run of whitespace (a tab, or more than one space), not just a single
    space, so a stray tab cannot fold a word into the keyword and drop it from
    the arguments a check below scans. Heredoc syntax (`<<EOF`) is refused
    outright: see HEREDOC above."""
    instructions: list[tuple[str, str]] = []
    pending = ""
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if HEREDOC.search(stripped):
            raise ValueError(f"heredoc syntax is not supported by this scanner: {stripped!r}")
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        parts = re.split(r"\s+", pending + stripped, maxsplit=1)
        pending = ""
        instructions.append((parts[0].upper(), parts[1].strip() if len(parts) > 1 else ""))
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


def _defined_stage_names(instructions: list[tuple[str, str]]) -> set[str]:
    """Every name a `--from=` may legitimately target: each stage's 0-based
    ordinal among FROM instructions, and its `AS name` alias when it has one."""
    names: set[str] = set()
    stage_number = 0
    for keyword, arguments in instructions:
        if keyword != "FROM":
            continue
        names.add(str(stage_number))
        words = arguments.split()
        if len(words) >= 3 and words[1].upper() == "AS":
            names.add(words[2])
        stage_number += 1
    return names


def _bad_from_flags(instructions: list[tuple[str, str]]) -> list[str]:
    """`--from=` targets on a COPY or RUN that name neither an earlier stage in
    this Dockerfile nor an entry in the fixed allowlist: an external image, or
    a stage that does not exist yet."""
    stages = _defined_stage_names(instructions)
    bad = []
    for keyword in ("COPY", "RUN"):
        for arguments in _all(instructions, keyword):
            parts = json.loads(arguments) if arguments.startswith("[") else shlex.split(arguments)
            for part in parts:
                if part.startswith("--from="):
                    target = part.split("=", 1)[1]
                    if target not in stages and target not in FROM_ALLOWLIST:
                        bad.append(target)
    return bad


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


@pytest.mark.parametrize("snippet", [
    "python -c \"load_model('legal-e5')\"",
    "curl -fsSL https://huggingface.co/intfloat/multilingual-e5-base/resolve/main/model.safetensors -o /tmp/m",
])
def test_build_time_download_pattern_catches_load_model_calls_and_the_huggingface_domain(snippet):
    assert BUILD_TIME_DOWNLOAD.search(snippet), snippet


def test_hf_hub_offline_is_set_as_a_build_arg_so_a_run_step_cannot_reach_the_network_for_a_model():
    args = _all(_instructions(_read("Dockerfile")), "ARG")

    assert "HF_HUB_OFFLINE=1" in args, args


def test_heredoc_syntax_in_a_run_is_refused():
    with pytest.raises(ValueError, match="heredoc"):
        _instructions("FROM x\nRUN <<EOF\necho hi\nEOF\n")
    with pytest.raises(ValueError, match="heredoc"):
        _instructions("FROM x\nRUN <<-EOF\n\techo hi\nEOF\n")


def test_instructions_split_on_any_whitespace_not_just_a_single_space():
    # A tab (or more than one space) between the keyword and its arguments must
    # not fold the next word into the keyword and drop it from the arguments a
    # check like _pip_installs scans.
    tabbed = _instructions("FROM x\nRUN\tpip install -c constraints.txt -r requirements.txt\n")
    spaced = _instructions("FROM x\nRUN   pip install -c constraints.txt -r requirements.txt\n")

    assert tabbed[-1] == ("RUN", "pip install -c constraints.txt -r requirements.txt")
    assert spaced[-1] == ("RUN", "pip install -c constraints.txt -r requirements.txt")


def test_the_dockerfile_has_no_from_flag_referencing_outside_its_own_stages():
    assert not _bad_from_flags(_instructions(_read("Dockerfile")))


def test_a_from_flag_referencing_an_external_image_is_flagged():
    instructions = _instructions(
        "FROM python:3.14-slim AS base\nFROM base\nCOPY --from=base /x /y\nCOPY --from=attacker/evil:latest /a /b\n")

    assert _bad_from_flags(instructions) == ["attacker/evil:latest"]


def test_a_from_flag_referencing_a_stage_by_its_ordinal_index_is_accepted():
    instructions = _instructions("FROM python:3.14-slim\nFROM debian\nCOPY --from=0 /x /y\n")

    assert not _bad_from_flags(instructions)


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


def _published_ports(service: dict) -> list[str]:
    """A service's `ports:`, short or long syntax, as `host_ip:published:target`
    strings. An explicit `/tcp` suffix (short syntax only; tcp is the default
    protocol either way) is stripped, so it names the same port as the bare form."""
    ports = []
    for p in service.get("ports") or []:
        text = f"{p.get('host_ip', '')}:{p.get('published', '')}:{p.get('target', '')}" if isinstance(p, dict) \
            else str(p)
        ports.append(re.sub(r"/tcp$", "", text, flags=re.IGNORECASE))
    return ports


def test_published_ports_treats_an_explicit_tcp_suffix_as_the_bare_port():
    assert _published_ports({"ports": ["127.0.0.1:8000:8000/tcp"]}) == ["127.0.0.1:8000:8000"]
    assert _published_ports({"ports": ["127.0.0.1:8000:8000"]}) == ["127.0.0.1:8000:8000"]


def test_the_port_is_published_on_the_hosts_loopback_only():
    app = _compose()["services"]["app"]
    ports = _published_ports(app)

    assert "127.0.0.1:8000:8000" in ports, ports
    assert all(port.startswith("127.0.0.1:") for port in ports), ports
    # Any other network mode, the host's above all, would bypass what is published.
    assert "network_mode" not in app


def test_expose_accepts_an_explicit_tcp_suffix():
    stage = _instructions("FROM x\nEXPOSE 8000/tcp\n")

    assert "8000" in {word.split("/")[0] for arguments in _all(stage, "EXPOSE") for word in arguments.split()}


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


def _is_pip_install(command: list[str]) -> bool:
    """True for `pip install`, `pip3 install`, `python -m pip install`, and the
    same forms invoked through a full path (`/opt/venv/bin/pip`, `/usr/bin/python3 -m pip`)."""
    program = posixpath.basename(command[0]) if command else ""
    return (program in ("pip", "pip3") and command[1:2] == ["install"]) or command[1:4] == ["-m", "pip", "install"]


def _pip_installs(stage: list[tuple[str, str]]) -> list[tuple[tuple[int, int], list[str]]]:
    """((instruction index, command index), words) for each pip install a RUN makes, in build order."""
    return [((i, j), command) for i, (keyword, arguments) in enumerate(stage) if keyword == "RUN"
            for j, command in enumerate(_commands(arguments)) if _is_pip_install(command)]


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


def test_the_cpu_torch_install_takes_no_deps_from_an_index_that_is_not_the_cpu_wheel_index():
    stage = _final_stage(_instructions(_read("Dockerfile")))
    from_cpu_index = [command for _, command in _pip_installs(stage) if any(CPU_INDEX in word for word in command)]

    assert from_cpu_index, "nothing installs torch from the CPU index"
    for command in from_cpu_index:
        assert "--no-deps" in command, f"torch's own resolver could still reach elsewhere: {' '.join(command)}"


def test_pip_installs_recognizes_a_full_path_or_module_invocation():
    stage = [("RUN", "/opt/venv/bin/pip install -c constraints.txt -r requirements.txt")]
    assert _pip_installs(stage), "a full-path pip invocation is not recognized"

    stage = [("RUN", "/usr/bin/python3.14 -m pip install -c constraints.txt -r requirements.txt")]
    assert _pip_installs(stage), "python -m pip through a full python path is not recognized"


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


UNCONFINED = {"seccomp:unconfined", "apparmor:unconfined"}


def _hardening_violations(name: str, service: dict) -> list[str]:
    """Every one of a compose service's worst options: privileged, the Docker
    socket bind-mounted in, a shared host pid/ipc namespace, an unconfined
    seccomp/apparmor profile, or an explicit override back to root."""
    violations = []
    if service.get("privileged"):
        violations.append(f"{name}: privileged undoes every restriction")
    if _mounts(service).get("/var/run/docker.sock", (None, None))[0] == "bind":
        violations.append(f"{name}: /var/run/docker.sock is bind-mounted in")
    if service.get("pid") == "host":
        violations.append(f"{name}: pid: host shares the host's process namespace")
    if service.get("ipc") == "host":
        violations.append(f"{name}: ipc: host shares the host's IPC namespace")
    security = {str(option).lower().replace("=", ":") for option in service.get("security_opt") or []}
    if security & UNCONFINED:
        violations.append(f"{name}: an unconfined seccomp/apparmor profile")
    if str(service.get("user", "")).lower() in ("root", "0"):
        violations.append(f"{name}: user: overrides back to root")
    return violations


@pytest.mark.parametrize("bad_service", [
    {"privileged": True},
    {"volumes": [{"type": "bind", "source": "/var/run/docker.sock", "target": "/var/run/docker.sock"}]},
    {"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]},
    {"pid": "host"},
    {"ipc": "host"},
    {"security_opt": ["seccomp:unconfined"]},
    {"security_opt": ["seccomp=unconfined"]},
    {"security_opt": ["apparmor:unconfined"]},
    {"user": "root"},
    {"user": "0"},
], ids=["privileged", "docker-sock-long", "docker-sock-short", "pid-host", "ipc-host",
        "seccomp-colon", "seccomp-equals", "apparmor", "user-root", "user-0"])
def test_each_dangerous_compose_option_is_caught(bad_service):
    assert _hardening_violations("svc", bad_service)


def test_no_compose_service_uses_a_dangerous_option():
    services = _compose().get("services") or {}

    assert services, "no service defined"
    for name, service in services.items():
        assert not _hardening_violations(name, service), _hardening_violations(name, service)


# ---------------------------------------------------------------------------
# the base image
# ---------------------------------------------------------------------------

# `sha256:` plus 64 lowercase hex, which is the only digest form a FROM takes.
_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _base_images(dockerfile: str) -> list[str]:
    """The image reference each FROM builds on, minus its flags and `AS name`."""
    bases = []
    for keyword, arguments in _instructions(dockerfile):
        if keyword != "FROM":
            continue
        words = [w for w in arguments.split() if not w.startswith("--")]
        bases.append(words[0])
    return bases


def test_every_base_image_is_pinned_to_a_digest():
    """A tag is a moving target: `python:3.14-slim` is rebuilt whenever a base package
    gets a CVE fix, so a rebuild of this file without a digest can quietly produce a
    different image from the one every other test in this file was written against.

    Nothing here builds, so this is the only place that failure could be caught.
    """
    bases = _base_images(_read("Dockerfile"))
    assert bases, "the Dockerfile has no FROM at all"
    unpinned = [b for b in bases if not _DIGEST.search(b)]
    assert not unpinned, f"base image(s) not pinned to a digest: {unpinned}"


def test_every_base_image_keeps_its_tag_beside_the_digest():
    """`python@sha256:cad9a2…` is valid and tells a reader nothing — not the language,
    not the version, not whether it is the slim variant. The tag is the documentation;
    the digest is what Docker actually resolves."""
    for base in _base_images(_read("Dockerfile")):
        name = base.split("@", 1)[0]
        assert ":" in name, f"{base} is pinned but names no tag"


@pytest.mark.parametrize("reference, pinned", [
    ("python:3.14-slim@sha256:" + "a" * 64, True),
    ("python:3.14-slim", False),
    ("python@sha256:" + "b" * 64, True),          # pinned, but the tag test above rejects it
    ("python:3.14-slim@sha256:" + "A" * 64, False),   # digests are lowercase hex
    ("python:3.14-slim@sha256:" + "c" * 63, False),   # one character short
    ("python:sha256-lookalike", False),
])
def test_the_digest_pattern_accepts_only_a_real_digest(reference, pinned):
    assert bool(_DIGEST.search(reference)) is pinned
