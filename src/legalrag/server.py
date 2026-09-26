"""Local test page for the retrieval system.

    python tasks.py serve            # http://127.0.0.1:8000

Deliberately a LOCAL server, not a hosted page: it reads the corpus and the
question set off disk, and the corpus is not redistributable. Standard library
only — no framework, no build step, nothing to install before it runs.

What the page is for, in order of importance:

1. **Run the eval set against the current retriever** and see which questions
   fail and why. This is the working surface of the project.
2. Ad-hoc query testing.
3. Browsing the ingested corpus to sanity-check the article split.

It is a development instrument, not a product demo. Numbers it shows are from
the dev split and are labelled as such.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .evaluate import corpus_size, load_questions
from .hosts import ALLOWED_HOSTS, split_host_header
from .retrieve import BM25Index, load_index, recall_at_k, reciprocal_rank

UI_PATH = Path("ui/index.html")

# A dev retrieval tool never needs more hits than this; also keeps a
# maliciously large k (e.g. from a rebound page) from being handed straight
# to the retriever.
_MAX_K = 50

_index: BM25Index | None = None


def _parse_k(qs: dict[str, list[str]]) -> int | None:
    """The `k` query parameter, bounded to [1, _MAX_K], or None if it is
    missing a valid value entirely (not a number, or out of bounds)."""
    raw = qs.get("k", ["5"])[0]
    try:
        k = int(raw)
    except ValueError:
        return None
    return k if 1 <= k <= _MAX_K else None


def index() -> BM25Index | None:
    global _index
    if _index is None:
        _index = load_index()
    return _index


def _status() -> dict:
    questions, errors = load_questions()
    idx = index()
    laws = sorted({d.get("law_name", "") for d in idx.docs}) if idx else []
    return {
        "corpus": corpus_size() or 0,
        "laws": [l for l in laws if l],
        "questions": len(questions),
        "unverified": sum(1 for q in questions if q.ref_status != "verified"),
        "errors": errors,
        "retriever": "BM25 (baseline)" if idx else None,
    }


def _questions() -> list[dict]:
    questions, _ = load_questions()
    return [q.__dict__ for q in questions]


def _search(q: str, k: int) -> list[dict]:
    idx = index()
    if not idx or not q:
        return []
    return [h.__dict__ for h in idx.search(q, k)]


def _eval_run(k: int) -> dict:
    """Run every answerable question through the retriever.

    Unanswerable (out_of_corpus) questions are reported separately: retrieval
    cannot be scored on them, they are an abstention test for M2's generator.
    """
    idx = index()
    questions, _ = load_questions()
    if not idx:
        return {"error": "corpus not ingested"}

    rows, recalls, rrs = [], [], []
    for q in questions:
        if not q.answerable:
            rows.append({
                "id": q.id, "category": q.category, "question": q.question,
                "expected": [], "hits": [], "recall": None, "rr": None,
                "note": "abstention test — not scored on retrieval",
            })
            continue
        hits = idx.search(q.question, k)
        r = recall_at_k(q.expected_articles, hits)
        rr = reciprocal_rank(q.expected_articles, hits)
        recalls.append(r)
        rrs.append(rr)
        rows.append({
            "id": q.id, "category": q.category, "question": q.question,
            "expected": q.expected_articles,
            "hits": [h.__dict__ for h in hits],
            "recall": round(r, 3), "rr": round(rr, 3),
            "ref_status": q.ref_status,
        })

    n = len(recalls)
    return {
        "k": k,
        "scored": n,
        "recall_at_k": round(sum(recalls) / n, 3) if n else None,
        "mrr": round(sum(rrs) / n, 3) if n else None,
        "rows": rows,
        "warning": "dev split · BM25 baseline · ground truth may still be unverified",
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload, status=200, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if split_host_header(self.headers.get("Host", ""))[0] not in ALLOWED_HOSTS:
            return self._send({"error": "invalid host"}, 400)

        url = urlparse(self.path)
        qs = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            if not UI_PATH.exists():
                return self._send(b"ui/index.html not found", 404, "text/plain; charset=utf-8")
            return self._send(UI_PATH.read_bytes(), 200, "text/html; charset=utf-8")
        if url.path == "/api/status":
            return self._send(_status())
        if url.path == "/api/questions":
            return self._send(_questions())
        if url.path == "/api/search":
            k = _parse_k(qs)
            if k is None:
                return self._send({"error": "invalid k"}, 400)
            return self._send(_search(qs.get("q", [""])[0], k))
        if url.path == "/api/eval":
            k = _parse_k(qs)
            if k is None:
                return self._send({"error": "invalid k"}, 400)
            return self._send(_eval_run(k))
        return self._send({"error": "not found"}, 404)

    def log_message(self, *args):  # keep the console readable
        pass


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    port = 8000
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])

    st = _status()
    print(f"corpus    : {st['corpus']} articles")
    print(f"questions : {st['questions']} ({st['unverified']} unverified)")
    if not st["corpus"]:
        print("\nNOTE: no corpus ingested — the page will load, search will be empty.")
    print(f"\n  http://127.0.0.1:{port}\n\nCtrl+C to stop.")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0
