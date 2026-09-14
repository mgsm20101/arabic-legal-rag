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

from dataclasses import dataclass

from .cite import ABSTAIN_MARKER
from .ollama import ollama_chat

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


def load_model(name: str = DEFAULT_MODEL, max_new_tokens: int = MAX_NEW_TOKENS):
    """A callable messages -> str, backed by transformers on CPU."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "transformers and torch are required to generate.\n"
            "  python tasks.py setup      (or: pip install -r requirements.txt)"
        ) from e

    tok = AutoTokenizer.from_pretrained(name)
    # No `device_map`: it makes `accelerate` a hard dependency and buys nothing,
    # because the torch here is a CPU-only build and loads to CPU by default.
    model = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE)
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


def resolve_model(spec: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """`spec` names the generation runtime: `hf:<repo>` loads a transformers
    checkpoint in-process via `load_model`; `ollama:<name>` talks to a local
    Ollama server instead via `ollama.ollama_chat` — the two runtimes ADR-023
    added. A single string is what a `--model` CLI flag and a saved run's
    `"model"` field both need to be: one value that round-trips between them.

    The part of `spec` after `ollama:` may itself contain a colon (Ollama tags
    look like `qwen3:4b`), so only the *first* colon separates the prefix.
    """
    prefix, _, rest = spec.partition(":")

    if prefix == "hf":
        if not rest.strip():
            raise ValueError(f"model spec {spec!r} names no repo after 'hf:'")
        return load_model(rest, max_new_tokens)

    if prefix == "ollama":
        name = rest
        if not name.strip():
            raise ValueError(f"model spec {spec!r} names no model after 'ollama:'")
        family = name.partition(":")[0]
        # qwen3 thinks by default, and unlike an ordinary reply, thinking
        # tokens would silently consume the 128-token cap and the time
        # budget before any citation is written. `think` is left unset for
        # every other family because its effect on a non-thinking model's
        # output is not documented.
        think = False if family == "qwen3" else None
        return ollama_chat(name, num_predict=max_new_tokens, think=think)

    raise ValueError(
        f"unknown model spec {spec!r} — expected 'hf:<repo>' or 'ollama:<name>'"
    )


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
