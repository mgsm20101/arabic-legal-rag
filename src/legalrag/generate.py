"""Citation-forced generation over retrieved articles — PRD M2.

The model is given the retrieved articles and asked for an answer in which
every claim carries the number of the article it rests on, or the fixed
abstention marker if the articles do not answer the question. Whether it obeys
is not assumed: `cite.audit` checks every answer against the corpus and against
the exact set of articles the retriever handed over, and `cite` has no model in
it, so the check holds regardless of what generated the text.

**Local, on CPU, by constraint not by preference.** `llama-cpp-python` has no
wheel for Python 3.14 at all, so the runtime here is `transformers` on the
torch that is already here — one of two runtimes `resolve_model` below now
chooses between; a local Ollama server is the other (ADR-023; ADR-024 will
hold the measured choice). That fixes the size: a 3B instruct model is what
a CPU answers 20 questions with in minutes rather than hours. The number
that matters for M2 is citation discipline, and a small model measures that
honestly — arguably more honestly, since a larger one hides grounding
failures behind fluency.

**The prompt asks for one citation form and the audit measures two.** The gap
between `[مادة N]` and every other way Arabic cites an article is the model's
instruction-following, reported as a number instead of an impression.

The model is injectable for the same reason as in `dense.py`: the test suite
must run on a fresh clone with no download and no torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .cite import ABSTAIN_MARKER
from .ollama import GeneratorUnavailable, ollama_chat

# Weights live inside the project (task 1.2), not only in the machine's HF
# cache — git-ignored, since they are too large to commit. The spec identity
# (`hf:Qwen/Qwen2.5-1.5B-Instruct`) does not change based on where the bytes
# happen to be; only `model_source` below does.
MODELS_DIR = Path(os.environ.get("LEGALRAG_MODELS_DIR", "models"))

# Measured, not chosen by size. Qwen2.5-3B-Instruct was tried first and is
# unusable here: its checkpoint is bfloat16, this CPU has no native bf16, so
# every matmul is emulated. One short answer took SIX HOURS end to end (load
# 6,465s, generate 15,499s) — timestamps in git history, not an estimate.
# The fix is not a smaller model, it is the dtype: float32 runs natively. A 3B
# in float32 needs 12.4 GB against 16 GB of RAM and would swap; 1.5B needs
# 6.2 GB and does not.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# float32, deliberately. See above: "auto" honours the checkpoint's bfloat16,
# which is correct on hardware that has it and catastrophic on hardware that
# does not. Nothing errors — it just runs four orders of magnitude too slow.
DTYPE = "float32"

# A citation-carrying legal answer of two or three sentences. Capped low
# because on CPU every token is wall-clock, and an answer that rambles past
# its citations is not more useful for what B1 measures.
MAX_NEW_TOKENS = 128

# Greedy. A temperature would make the citation numbers themselves vary between
# runs, and B1 is a claim about the system, not about one sample of it. Used
# below as `do_sample = TEMPERATURE > 0`. `ollama.py` fixes the same 0.0/seed
# 0 as its own literal rather than importing this one: `ollama_chat` is
# imported from there into here, so the reverse import would be a cycle.
TEMPERATURE = 0.0

SYSTEM = f"""أنت مساعد قانوني. تجيب **فقط** من نصوص المواد المعطاة لك أدناه.

القواعد:
1. كل جملة تحمل حكماً يجب أن تنتهي بمرجعها بالصيغة: [مادة رقم]
2. لا تستشهد بأي مادة غير المواد المعطاة لك. لا تعتمد على معرفتك السابقة.
3. إذا كانت المواد المعطاة لا تجيب على السؤال، اكتب هذه الجملة وحدها ولا شيء غيرها:
{ABSTAIN_MARKER}
4. أجب بالعربية، جملتين أو ثلاثاً على الأكثر. لا تشرح القواعد ولا تعتذر."""

USER = """المواد المتاحة:

{articles}

السؤال: {question}"""


@dataclass
class Answer:
    question: str
    text: str
    article_numbers: list[int]   # what the retriever supplied, in rank order


def format_articles(articles: list[dict]) -> str:
    """`articles` are dicts with `number` and `text`, in retrieval rank order."""
    return "\n\n".join(f"[مادة {a['number']}]\n{a['text']}" for a in articles)


def build_messages(question: str, articles: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER.format(
            articles=format_articles(articles), question=question.strip())},
    ]


def _looks_like_a_checkpoint(path: Path) -> bool:
    """`config.json` alone does not prove real weights are there — a
    half-copied directory (or one made by mistake) is not a usable
    checkpoint, and trusting it as one would fail confusingly deep inside
    transformers instead of falling back to the hub cleanly."""
    if not (path / "config.json").exists():
        return False
    return (
        any(path.glob("*.safetensors"))
        or any(path.glob("*.bin"))
        or (path / "tokenizer.json").exists()
    )


def model_source(name: str, models_dir: Path | None = None) -> str:
    """Where to actually read `name`'s weights from.

    Local-first: `name` itself may already be a directory (an explicit path,
    trusted as given); otherwise `models_dir/<name>` is tried in full —
    `"Qwen/Qwen2.5-1.5B-Instruct"` -> `models/Qwen/Qwen2.5-1.5B-Instruct` —
    and used only if it looks like a real checkpoint. The org segment is not
    optional: `someorg/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`
    share a bare repo name but are different weights, and collapsing the
    path to just the name would silently hand one org's request the other's
    bytes the moment both happened to be cached locally. Falling through to
    `name` unchanged lets transformers resolve it from the HF hub cache,
    exactly as before this function existed — a repo not yet copied into
    `models/` still works.
    """
    d = MODELS_DIR if models_dir is None else models_dir
    if Path(name).is_dir():
        return name
    candidate = d / name
    if _looks_like_a_checkpoint(candidate):
        return str(candidate)
    return name


def load_model(name: str = DEFAULT_MODEL, max_new_tokens: int = MAX_NEW_TOKENS):
    """A callable messages -> str, backed by transformers on CPU."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        # Not SystemExit: this can run inside a server process (the P1 demo
        # layer, ADR-023), and library code exiting the process out from
        # under a caller is exactly what GeneratorUnavailable exists to
        # avoid — see ollama.py, which never raises SystemExit either.
        raise GeneratorUnavailable(
            "transformers and torch are required to generate.\n"
            "  python tasks.py setup      (or: pip install -r requirements.txt)"
        ) from e

    source = model_source(name)
    try:
        tok = AutoTokenizer.from_pretrained(source)
        # No `device_map`: it makes `accelerate` a hard dependency and buys
        # nothing, because the torch here is a CPU-only build and loads to
        # CPU by default.
        model = AutoModelForCausalLM.from_pretrained(source, dtype=DTYPE)
    except (OSError, ValueError) as e:
        # A missing/incomplete local checkpoint or an unresolvable hub id —
        # an environment problem the caller (answer_eval.main) should be
        # able to catch and report, same as an unreachable Ollama server.
        raise GeneratorUnavailable(f"could not load weights from {source}: {e}") from e
    model.eval()

    def run(messages: list[dict]) -> str:
        import torch

        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([prompt], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=TEMPERATURE > 0, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    return run


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split a `--model` spec into `(runtime, name)`, validating as it goes —
    `("hf", "Qwen/Qwen2.5-1.5B-Instruct")`, `("ollama", "qwen3:4b")` — without
    loading a checkpoint, importing torch, or making a network call.

    Cheap enough to call as the very first thing `answer_eval.main` does
    with a `--model` value, before loading the corpus or the question set:
    a typo in the spec should not cost either.

    The part of `spec` after `ollama:` may itself contain a colon (Ollama
    tags look like `qwen3:4b`), so only the *first* colon separates the
    prefix.
    """
    prefix, _, rest = spec.partition(":")

    if prefix == "hf":
        if not rest.strip():
            raise ValueError(f"model spec {spec!r} names no repo after 'hf:'")
        return "hf", rest

    if prefix == "ollama":
        if not rest.strip():
            raise ValueError(f"model spec {spec!r} names no model after 'ollama:'")
        return "ollama", rest

    raise ValueError(
        f"unknown model spec {spec!r} — expected 'hf:<repo>' or 'ollama:<name>'"
    )


def resolve_model(spec: str, max_new_tokens: int = MAX_NEW_TOKENS, fmt: dict | None = None):
    """`spec` names the generation runtime: `hf:<repo>` loads a transformers
    checkpoint in-process via `load_model`; `ollama:<name>` talks to a local
    Ollama server instead via `ollama.ollama_chat` — the two runtimes ADR-023
    added. A single string is what a `--model` CLI flag and a saved run's
    `"model"` field both need to be: one value that round-trips between them.
    Parsing and validating `spec` itself is `parse_model_spec`'s job; this
    is only the dispatch on top of it.

    `fmt` is Run 5's schema-constrained decoding request
    (`claims.CLAIMS_SCHEMA`) — meaningful only for the `ollama:` runtime,
    the only one of the two with any such feature. Passing it for `hf:` is
    rejected outright rather than silently ignored: a caller that thinks it
    asked for constrained JSON and got free text back would have no way to
    find out short of parsing failures downstream, well after the model
    already ran.
    """
    prefix, name = parse_model_spec(spec)

    if prefix == "hf":
        if fmt is not None:
            raise ValueError(
                f"model spec {spec!r} is hf:, but the claims contract needs "
                "an ollama: model — schema-constrained decoding is not "
                "available on the transformers path here"
            )
        return load_model(name, max_new_tokens)

    family = name.partition(":")[0]
    # Measured, not assumed (EVAL.md, ADR-024): on Ollama 0.20.3, `think:
    # false` does not stop qwen3 from thinking — it moves the thinking into
    # the visible answer text, at the same token count as `think: true`, and
    # `/no_think` in the prompt was ignored outright (533 chars of thinking,
    # empty answer). qwen3:4b was excluded from Run 4 for exactly this. The
    # flag is still sent for the qwen3 family, unset for every other one,
    # so a future qwen3 run starts from the documented request rather than
    # from silence — the underlying behaviour is Ollama's to fix, not this
    # client's to work around.
    think = False if family == "qwen3" else None
    return ollama_chat(name, num_predict=max_new_tokens, think=think, fmt=fmt)


class Generator:
    """Answers a question from a ranked list of articles."""

    def __init__(self, model=None, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_name)
        return self._model

    def answer(self, question: str, articles: list[dict]) -> Answer:
        if not articles:
            # Nothing retrieved is not a hard question — it is no context at
            # all, and answering from an empty context is exactly the failure
            # the abstention rule exists for.
            return Answer(question, ABSTAIN_MARKER, [])
        text = self.model(build_messages(question, articles))
        return Answer(question, text.strip(), [a["number"] for a in articles])
