"""The reported failure, reproduced end to end against the container.

A user opens the page the moment the app answers and asks a question, on a
machine that has never downloaded the encoder. Before ADR-031 that request
spent the whole 1.1 GB download inside `ask`'s budget and came back 429 with
«waited 244.9s for a generation slot» while nothing was queued.

Two things are watched at once, because the fix has two halves: the question
must be answered, and `/` -- the container's HEALTHCHECK path -- must keep
answering throughout, since warming runs in a background thread for exactly
that reason.
"""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).with_name("coldstart_result.json")
health_samples, stop = [], threading.Event()


def watch_health():
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(f"{BASE}/", timeout=5) as r:
                health_samples.append((round(time.perf_counter() - t0, 3), r.status))
        except Exception as e:                       # noqa: BLE001 - recorded, not raised
            health_samples.append((round(time.perf_counter() - t0, 3), type(e).__name__))
        stop.wait(2.0)


# Wait only for the socket to open -- that is the instant a user could ask.
t_up = time.perf_counter()
while time.perf_counter() - t_up < 120:
    try:
        urllib.request.urlopen(f"{BASE}/", timeout=3)
        break
    except Exception:                                # noqa: BLE001
        time.sleep(0.2)
up_s = time.perf_counter() - t_up
print(f"app answered / after {up_s:.1f}s; asking immediately", flush=True)

threading.Thread(target=watch_health, daemon=True).start()

body = json.dumps({"question": "ما المدة التي يلتزم فيها المتحكم بالبت في الطلب؟"}).encode()
req = urllib.request.Request(f"{BASE}/api/chat", data=body, method="POST",
                             headers={"Content-Type": "application/json", "X-LegalRAG": "1"})
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        code, out = r.status, json.loads(r.read())
except urllib.error.HTTPError as e:
    code, out = e.code, json.loads(e.read())
elapsed = time.perf_counter() - t0
stop.set()
time.sleep(0.1)

slow = [s for s in health_samples if isinstance(s[1], int) and s[0] > 1.0]
bad = [s for s in health_samples if not isinstance(s[1], int) or s[1] != 200]
print(f"chat: HTTP {code} after {elapsed:.1f}s", flush=True)
print(f"health probes while it ran: {len(health_samples)}, "
      f"non-200: {len(bad)}, slower than 1s: {len(slow)}", flush=True)
OUT.write_text(json.dumps({
    "socket_open_after_s": round(up_s, 1), "chat_http": code,
    "chat_elapsed_s": round(elapsed, 1), "status": out.get("status"),
    "claims": out.get("claims"), "timings_ms": out.get("timings_ms"),
    "error": out.get("error"), "message_ar": out.get("message_ar"),
    "health_probes": len(health_samples), "health_non_200": bad,
    "health_slowest_s": max((s[0] for s in health_samples), default=None),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("written", OUT.name, flush=True)
