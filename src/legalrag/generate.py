"""Model specs and loading for both products, plus the legacy text contract.

`parse_model_spec` and `resolve_model` turn ``hf:<repo>`` or ``ollama:<name>``
into a ``messages -> str`` callable; the app and every answer-eval contract use
them. `Generator` answers in free text with inline ``[مادة N]`` citations,
checked afterwards by `cite.audit`. It and the ``transformers`` path are legacy,
kept to reproduce the E3 runs in EVIDENCE.md; the app uses `claims` instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .cite import ABSTAIN_MARKER
from .ollama import GeneratorUnavailable, ollama_chat

# Local checkpoints (git-ignored); the spec names the model, not where its bytes are.
MODELS_DIR = Path(os.environ.get("LEGALRAG_MODELS_DIR", "models"))

# 1.5B in float32 fits 16 GB of RAM; the 3B bf16 checkpoint took six hours per answer on CPU.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Not "auto": emulated bfloat16 on a CPU without it is four orders of magnitude slower.
DTYPE = "float32"

# Two or three cited sentences; on CPU every token is wall-clock.
MAX_NEW_TOKENS = 128

# Greedy, so citations do not vary between runs. ollama.py repeats the value to avoid an import cycle.
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
    """A directory with `config.json` and actual weights or a tokenizer, not a half-copied one."""
    if not (path / "config.json").exists():
        return False
    return (
        any(path.glob("*.safetensors"))
        or any(path.glob("*.bin"))
        or (path / "tokenizer.json").exists()
    )


def model_source(name: str, models_dir: Path | None = None) -> str:
    """Where to actually read `name`'s weights from.

    `name` itself if it is a directory, else `models_dir/<org>/<name>` if it holds
    a real checkpoint (the org is kept: two orgs can share a repo name), else
    `name` unchanged for transformers to resolve from the hub cache.
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
        # Not SystemExit: this may run inside a server process.
        raise GeneratorUnavailable(
            "transformers and torch are required to generate.\n"
            "  python tasks.py setup      (or: pip install -r requirements.txt)"
        ) from e

    source = model_source(name)
    try:
        tok = AutoTokenizer.from_pretrained(source)
        # No device_map: it would require accelerate, and CPU is the default anyway.
        model = AutoModelForCausalLM.from_pretrained(source, dtype=DTYPE)
    except (OSError, ValueError) as e:
        # An environment problem the caller reports, like an unreachable Ollama.
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

    Only the first colon splits: Ollama tags contain colons (`qwen3:4b`).
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
    """A `messages -> str` model for `spec`: `hf:` loads transformers in-process,
    `ollama:` talks to a local Ollama server.

    `fmt` requests schema-constrained JSON (`claims.CLAIMS_SCHEMA`). Only Ollama
    supports it, so passing it with `hf:` is a ValueError, not silently ignored.
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
    # qwen3 is sent think=false; Ollama 0.20.3 ignores it (see DECISIONS.md ADR-024).
    think = False if family == "qwen3" else None
    return ollama_chat(name, num_predict=max_new_tokens, think=think, fmt=fmt)


class Generator:
    """Legacy free-text answerer (the `text` contract), kept to reproduce E3."""

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
            # No context: abstain without asking the model.
            return Answer(question, ABSTAIN_MARKER, [])
        text = self.model(build_messages(question, articles))
        return Answer(question, text.strip(), [a["number"] for a in articles])
