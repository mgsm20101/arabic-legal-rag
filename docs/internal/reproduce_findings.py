"""Read-only audit probes; generated documents and indexes use temporary directories.

Run from the repository root: python docs/internal/reproduce_findings.py
These probes describe current behavior, not the desired regression-test contract.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from legalrag import cite, ingest, ocr_gate, pdf_text
from legalrag.claims import ClaimsGenerator, parse_claims
from legalrag.dense import DenseIndex
from legalrag.library import Library
from legalrag.server import Handler


class CountingEncoder:
    def __init__(self):
        self.queries = 0

    def encode(self, texts, **kwargs):
        import numpy as np
        self.queries += sum(t.startswith("query: ") for t in texts)
        return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


def main():
    observations = {}
    contradictory = {"abstain": False, "claims": [
        {"text": "مدة تقديم الطلب تسعون يوماً.", "sources": [1]}
    ]}
    verdict = cite.gate(contradictory, [None], ["مدة تقديم الطلب ثلاثون يوماً."])
    observations["citation_is_not_entailment"] = {
        "status": verdict["status"], "kept": verdict["kept"]
    }

    gt = {
        "p01": {"tokens": ["12"], "critical": {"12": "header"}},
        "p02": {"tokens": ["99"], "critical": {"99": "header"}},
    }
    observations["ocr_missing_page"] = ocr_gate.run(gt, {"p01": "12"})
    observations["ocr_spurious_digits"] = ocr_gate.run(gt, {"p01": "12 777 888", "p02": "99"})

    with tempfile.TemporaryDirectory(prefix="legalrag-audit-") as directory:
        root = Path(directory)
        raw = root / "raw"
        raw.mkdir()
        # Arabic text is long enough to pass extraction; article 2 is missing.
        text = "مادة 1\n" + "هذا نص تجريبي للتحقق من سلامة المستند. " * 5
        text += "\nمادة 3\n" + "هذا نص تجريبي آخر للتحقق من سلامة المستند. " * 5
        (raw / "invalid.txt").write_text(text, encoding="utf-8")
        output = root / "articles.jsonl"
        output.write_text("PREVIOUS_VALID_CORPUS", encoding="utf-8")
        old_raw, old_out = ingest.RAW_DIR, ingest.OUT_PATH
        try:
            ingest.RAW_DIR, ingest.OUT_PATH = raw, output
            with contextlib.redirect_stdout(io.StringIO()):
                code = ingest.main(["--law", "قانون تجريبي"])
        finally:
            ingest.RAW_DIR, ingest.OUT_PATH = old_raw, old_out
        observations["failed_ingest_overwrites_previous_corpus"] = {
            "exit_code": code,
            "previous_corpus_preserved": output.read_text(encoding="utf-8") == "PREVIOUS_VALID_CORPUS",
        }

        encoder = CountingEncoder()
        library = Library(root / "library", encoder=encoder)
        for name in ("first", "second", "third"):
            library.add((name + " " + "هذا مستند تجريبي طويل بما يكفي للفهرسة. " * 10).encode(), name + ".txt")
        encoder.queries = 0
        library.search("سؤال تجريبي")
        observations["query_embeddings_per_search"] = {
            "documents": 3, "encoder_query_calls": encoder.queries
        }

        import numpy as np
        docs = [{"id": "law-1", "number": 1, "text": "test passage"}]
        cache = root / "cache.npz"
        index = DenseIndex(docs, encoder=encoder, cache_path=cache)
        index.save()
        with np.load(cache, allow_pickle=False) as saved:
            meta = saved["meta"].copy()
        np.savez(cache, embeddings=np.zeros((0, 2)), meta=meta)
        damaged = DenseIndex(docs, encoder=encoder, cache_path=cache)
        observations["valid_metadata_empty_embedding_matrix"] = {
            "accepted_as_cache": damaged.from_cache,
            "hits": len(damaged.search("query")),
        }

    observations["latin_only_line_with_keep_latin"] = pdf_text.logical_line(
        [{"text": "VPN", "x0": 0, "x1": 10}], keep_latin=True
    )
    parsed = parse_claims(json.dumps({"abstain": False, "claims": [
        {"text": "", "sources": [1]} for _ in range(4)
    ]}))
    observations["schema_accepts_four_empty_claims"] = len(parsed["claims"])

    class TruncatedReply:
        def __init__(self):
            self.calls = []

        def __call__(self, messages):
            self.calls.append({"truncated": True, "cut": True})
            return json.dumps(contradictory)

    answer = ClaimsGenerator(TruncatedReply()).answer("ما مدة تقديم الطلب؟", ["مدة تقديم الطلب ثلاثون يوماً."])
    observations["truncated_reply_accepted"] = {
        "schema_failure": answer.schema_failure,
        "gate_status": cite.gate(answer.parsed, [None], ["مدة تقديم الطلب ثلاثون يوماً."])["status"],
        "calls": answer.calls,
    }

    # Only connects to a disposable loopback listener. No corpus content is requested.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request("GET", "/api/questions", headers={"Host": "untrusted.invalid"})
        response = connection.getresponse()
        observations["legacy_server_foreign_host"] = {"status": response.status, "body_bytes": len(response.read())}
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    print(json.dumps(observations, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
