"""scripts/make_constraints.py, and the constraints.txt it wrote.

The generator is tested on fake installed distributions, so these tests read
nothing from the machine they run on; CI's environment is not the one the
file was generated in. The last test reads the committed files: every
requirement in every requirements file must admit its constraint's version, or
pip refuses the install, which only a Docker build or a CI run would show.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT_FILES = ("requirements.txt", "requirements-ci.txt", "requirements-dev.txt")
_spec = importlib.util.spec_from_file_location("make_constraints", ROOT / "scripts" / "make_constraints.py")
make_constraints = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("make_constraints", make_constraints)
_spec.loader.exec_module(make_constraints)

LINUX = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix"}
WINDOWS = {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt"}


class _Dist:
    def __init__(self, name: str, version: str, *requires: str):
        self.metadata = {"Name": name}
        self.version = version
        self.requires = list(requires) or None  # importlib.metadata's None for no Requires-Dist


def _installed(*dists: _Dist):
    by_name = {canonicalize_name(d.metadata["Name"]): d for d in dists}

    def distribution(name: str) -> _Dist:
        try:
            return by_name[canonicalize_name(name)]
        except KeyError:
            raise importlib.metadata.PackageNotFoundError(name) from None

    return distribution


def test_a_local_version_label_is_dropped_so_the_pin_matches_the_cpu_wheel():
    result = make_constraints.closure(["torch>=2"], [LINUX], _installed(_Dist("torch", "2.14.0+cpu")))

    assert result.pins == {"torch": "2.14.0"}


def test_a_dependency_is_kept_when_its_marker_holds_on_linux_or_on_this_machine():
    installed = _installed(
        _Dist("app", "1.0", 'on-linux; sys_platform == "linux"', 'on-windows; sys_platform == "win32"',
              'on-mac; sys_platform == "darwin"'),
        _Dist("on-linux", "1"), _Dist("on-windows", "2"), _Dist("on-mac", "3"))

    result = make_constraints.closure(["app"], [LINUX, WINDOWS], installed)

    assert result.pins == {"app": "1.0", "on-linux": "1", "on-windows": "2"}


def test_extras_are_followed_whether_a_requirement_file_or_a_dependency_asks_for_them():
    installed = _installed(
        _Dist("server", "1", 'speedups; extra == "fast"', 'theme; extra == "docs"', "codec[zstd]"),
        _Dist("speedups", "2"), _Dist("theme", "3"),
        _Dist("codec", "4", 'zstandard; extra == "zstd"'), _Dist("zstandard", "5"))

    result = make_constraints.closure(["server[fast]"], [LINUX], installed)

    assert result.pins == {"server": "1", "speedups": "2", "codec": "4", "zstandard": "5"}


def test_a_package_reached_again_with_another_extra_gains_that_extras_dependencies():
    installed = _installed(_Dist("lib", "1", 'a; extra == "x"', 'b; extra == "y"'), _Dist("a", "1"),
                           _Dist("b", "1"), _Dist("other", "1", "lib[y]"))

    result = make_constraints.closure(["lib[x]", "other"], [LINUX], installed)

    assert set(result.pins) == {"lib", "a", "b", "other"}


def test_a_reachable_package_that_is_not_installed_is_reported_and_left_unpinned():
    installed = _installed(_Dist("app", "1", "ghost>=1", 'only-on-mac; sys_platform == "darwin"'))

    result = make_constraints.closure(["app"], [LINUX], installed)

    assert result.pins == {"app": "1"}
    assert result.missing == ["ghost"]


def test_an_installed_version_outside_a_required_range_is_reported_as_a_conflict():
    result = make_constraints.closure(["app"], [LINUX], _installed(_Dist("app", "1", "lib>=2"), _Dist("lib", "1.5")))

    assert len(result.conflicts) == 1 and "lib>=2" in result.conflicts[0] and "1.5" in result.conflicts[0]


def test_requirement_files_are_read_through_their_r_includes_without_options_or_comments(tmp_path):
    (tmp_path / "base.txt").write_text("numpy>=1.24  # a comment\n\n# only a comment\n", encoding="utf-8")
    (tmp_path / "dev.txt").write_text("-r base.txt\n--index-url https://example.invalid/simple\npytest==9.1.1\n",
                                      encoding="utf-8")

    assert make_constraints.read_requirements(tmp_path / "dev.txt") == ["numpy>=1.24", "pytest==9.1.1"]


def test_requirements_dev_is_one_of_the_generators_roots():
    # tasks.py and the Makefile install requirements-dev.txt under constraints.txt
    # too, so anything it alone needs (pytest's own dependencies, say) must be
    # reachable from here as well, not just incidentally pinned by another root.
    assert "requirements-dev.txt" in make_constraints.ROOTS


@pytest.mark.parametrize("line", [
    "-e git+https://github.com/example/pkg.git#egg=pkg",
    "--editable git+https://github.com/example/pkg.git#egg=pkg",
    "--editable=git+https://github.com/example/pkg.git#egg=pkg",
])
def test_an_editable_or_vcs_requirement_line_fails_loudly_instead_of_being_silently_dropped(tmp_path, line):
    (tmp_path / "req.txt").write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        make_constraints.read_requirements(tmp_path / "req.txt")


def test_a_direct_url_requirement_fails_loudly_instead_of_pinning_whatever_is_installed(tmp_path):
    (tmp_path / "req.txt").write_text("pkg @ https://example.invalid/pkg-1.0-py3-none-any.whl\n", encoding="utf-8")

    with pytest.raises(ValueError):
        make_constraints.read_requirements(tmp_path / "req.txt")


def test_an_unparsable_requirement_line_fails_loudly(tmp_path):
    (tmp_path / "req.txt").write_text(">=1.0\n", encoding="utf-8")  # an operator with no package name

    with pytest.raises(ValueError):
        make_constraints.read_requirements(tmp_path / "req.txt")


def test_the_file_is_its_header_then_one_pin_per_line_sorted_by_canonical_name():
    text = make_constraints.render({"PyYAML": "6.0.3", "numpy": "2.5.1", "Jinja2": "3.1.6"}, header="# made here")

    assert text.splitlines() == ["# made here", "Jinja2==3.1.6", "numpy==2.5.1", "PyYAML==6.0.3"]


def _pins(path: Path) -> dict[str, str]:
    pairs = [line.split("==") for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    return {canonicalize_name(name): version for name, version in pairs}


def test_the_committed_constraints_satisfy_every_requirement_in_every_requirements_file():
    pins = _pins(ROOT / "constraints.txt")

    for file in REQUIREMENT_FILES:
        for line in make_constraints.read_requirements(ROOT / file):
            requirement = Requirement(line)
            name = canonicalize_name(requirement.name)
            # A constraint outside the requirement's range makes pip refuse the install,
            # which only a Docker build or a CI run would show.
            assert name in pins and requirement.specifier.contains(pins[name], prereleases=True), f"{file}: {line}"
    assert not [version for version in pins.values() if "+" in version], "a local label would never match a PyPI wheel"
    assert pins["torch"] == "2.14.0"
