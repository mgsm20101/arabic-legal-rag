"""Replace a file so a reader sees the old version or the new one, never half of either."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path


def write_atomic(path: Path, write: Callable[[Path], None], suffix: str = ".tmp") -> None:
    """Call `write(tmp)` on a temp file beside `path`, then os.replace it over `path`.

    `path` is never opened, so a failure leaves it untouched. `suffix` ends the
    temp name: np.savez appends `.npz` to any path that lacks it.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}{suffix}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, data: dict | list) -> None:
    """`data` as indented UTF-8 JSON at `path`, written atomically."""
    text = json.dumps(data, ensure_ascii=False, indent=1)
    write_atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
