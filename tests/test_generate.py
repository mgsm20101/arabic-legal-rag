"""Generation tests — PRD M2. No model, no download, no torch.

A stub model stands in for the real one, the same way `test_dense.py` uses a
stub encoder: a fresh clone must run the suite green in a second. What is being
tested here is the wiring, and the wiring is where the silent failures live —
a prompt that omits the article numbers, or an audit run against the corpus
instead of against what was actually retrieved, both produce plausible output
and a meaningless number.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.cite import ABSTAIN_MARKER, audit  # noqa: E402
from legalrag.generate import (  # noqa: E402
    SYSTEM,
    Generator,
    build_messages,
    format_articles,
    resolve_model,
)
from legalrag.ollama import OllamaChat  # noqa: E402

ARTICLES = [
    {"number": 7, "text": "يلتزم المتحكم بإبلاغ المركز خلال اثنتين وسبعين ساعة."},
    {"number": 12, "text": "يجوز للمركز الإعفاء من الإخطار في حالات محددة."},
]


class StubModel:
    """Returns a canned answer and records what it was asked."""

    def __init__(self, reply: str):
        self.reply = reply
        self.messages = None

    def __call__(self, messages):
        self.messages = messages
        return self.reply


def test_the_article_numbers_reach_the_prompt():
    """Without them the model cannot cite what it was given, and every answer
    is ungrounded by construction — with no error anywhere."""
    blob = format_articles(ARTICLES)
    assert "[مادة 7]" in blob and "[مادة 12]" in blob
    assert "اثنتين وسبعين" in blob


def test_the_full_article_text_is_sent_not_a_snippet():
    for a in ARTICLES:
        assert a["text"] in format_articles(ARTICLES)


def test_the_prompt_states_the_abstention_marker_verbatim():
    """The audit matches this string exactly. If the prompt asks for different
    words than the checker looks for, every abstention scores as a failure."""
    assert ABSTAIN_MARKER in SYSTEM


def test_the_question_reaches_the_prompt():
    msgs = build_messages("كم مهلة الإبلاغ؟", ARTICLES)
    assert msgs[0]["role"] == "system"
    assert "كم مهلة الإبلاغ؟" in msgs[1]["content"]


def test_an_answer_carries_the_ids_that_were_actually_retrieved():
    """The audit needs the retrieved set, not the corpus, to tell a recalled
    article from a read one."""
    g = Generator(model=StubModel("الإبلاغ خلال 72 ساعة [مادة 7]."))
    a = g.answer("كم المهلة؟", ARTICLES)
    assert a.article_numbers == [7, 12]


def test_an_empty_retrieval_abstains_without_calling_the_model():
    """No context is not a hard question — answering from nothing is the exact
    failure the abstention rule exists for."""
    stub = StubModel("لن يُستدعى")
    a = Generator(model=stub).answer("سؤال", [])
    assert a.text == ABSTAIN_MARKER
    assert stub.messages is None


def test_a_grounded_stub_answer_passes_the_audit_end_to_end():
    g = Generator(model=StubModel("يلتزم المتحكم بالإبلاغ خلال المهلة المقررة [مادة 7]."))
    a = g.answer("كم المهلة؟", ARTICLES)
    r = audit(a.text, set(range(1, 50)), set(a.article_numbers))
    assert r["grounded"] is True


def test_a_recalled_article_fails_the_audit_end_to_end():
    """Article 30 exists in the law and was not retrieved. Any check that only
    asks whether the number exists would pass this."""
    g = Generator(model=StubModel("يلتزم المعالج بحفظ السجلات لمدة سنة كاملة [مادة 30]."))
    a = g.answer("كم المهلة؟", ARTICLES)
    r = audit(a.text, set(range(1, 50)), set(a.article_numbers))
    assert r["ungrounded"] == [30]
    assert r["grounded"] is False


def test_a_model_spec_picks_the_runtime_without_loading_anything(monkeypatch):
    """Run 4 changes only the model; `resolve_model` is the switch. Deciding
    which runtime a spec means must not itself load a checkpoint or touch the
    network — the test suite has to stay model-free and offline."""
    import legalrag.generate as generate_mod

    calls = []

    def fake_load_model(name, max_new_tokens):
        calls.append((name, max_new_tokens))
        return "STUB-HF-MODEL"

    monkeypatch.setattr(generate_mod, "load_model", fake_load_model)

    hf_model = resolve_model("hf:Qwen/Qwen2.5-1.5B-Instruct")
    assert hf_model == "STUB-HF-MODEL"
    assert calls == [("Qwen/Qwen2.5-1.5B-Instruct", generate_mod.MAX_NEW_TOKENS)]

    # qwen3 thinks by default; unchecked, its thinking tokens would silently
    # eat the 128-token cap before a citation is ever written.
    qwen3 = resolve_model("ollama:qwen3:4b")
    assert isinstance(qwen3, OllamaChat)
    assert qwen3.model == "qwen3:4b"
    assert qwen3.think is False

    # Not qwen3 — `think` is left unset, since its effect on other model
    # families' output is not documented.
    qwen25 = resolve_model("ollama:qwen2.5:7b-instruct")
    assert isinstance(qwen25, OllamaChat)
    assert qwen25.model == "qwen2.5:7b-instruct"
    assert qwen25.think is None

    with pytest.raises(ValueError):
        resolve_model("bogus:x")
