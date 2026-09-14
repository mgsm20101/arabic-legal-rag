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
    model_source,
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
    # `ollama_chat` takes no default num_predict on purpose (a silent one
    # would silently borrow this project's own cap for an unrelated
    # caller) — resolve_model must still pass the real one through.
    assert qwen3.num_predict == generate_mod.MAX_NEW_TOKENS

    # Not qwen3 — `think` is left unset, since its effect on other model
    # families' output is not documented.
    qwen25 = resolve_model("ollama:qwen2.5:7b-instruct")
    assert isinstance(qwen25, OllamaChat)
    assert qwen25.model == "qwen2.5:7b-instruct"
    assert qwen25.think is None

    with pytest.raises(ValueError):
        resolve_model("bogus:x")


def test_an_empty_model_name_is_rejected_not_silently_loaded(monkeypatch):
    """`hf`, `hf:`, `ollama`, `ollama:` and a whitespace-only name after the
    colon all name no model at all. Before this check, `hf:` fell through to
    `load_model("", ...)` and `ollama:` to an Ollama request for a model
    literally named the empty string — both nonsensical, neither an error.

    `load_model` is stubbed so a spec that (wrongly) passes validation cannot
    fall through to a real transformers/network call from a test.
    """
    import legalrag.generate as generate_mod

    monkeypatch.setattr(generate_mod, "load_model", lambda name, n: "STUB-HF-MODEL")

    for spec in ("hf", "hf:", "hf:   ", "ollama", "ollama:", "ollama:   "):
        with pytest.raises(ValueError):
            resolve_model(spec)


def test_a_model_present_under_models_dir_loads_from_disk_not_the_hub(tmp_path):
    """Weights now live inside the project under models/ (task 1.2) —
    when a repo's files are there, `load_model` must read them from disk
    instead of going to the HF hub cache."""
    model_dir = tmp_path / "Qwen2.5-1.5B-Instruct"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    source = model_source("Qwen/Qwen2.5-1.5B-Instruct", models_dir=tmp_path)
    assert source == str(model_dir)


def test_a_model_absent_locally_falls_back_to_its_hub_id(tmp_path):
    """No local copy under models/ — the hub id is returned unchanged, so
    transformers resolves it from the HF cache exactly as before this
    feature existed. `tmp_path` is empty, so nothing is found there."""
    source = model_source("Qwen/Qwen2.5-1.5B-Instruct", models_dir=tmp_path)
    assert source == "Qwen/Qwen2.5-1.5B-Instruct"


def test_an_explicit_directory_is_used_as_given(tmp_path):
    """A caller that already names a directory (not a hub id) is trusted
    outright — no models_dir guessing games layered on top of an explicit
    path, and no config.json required at that path either.

    A same-named directory that *does* have config.json is planted under
    models_dir on purpose: without the `Path(name).is_dir()` branch, the
    fallback would find that shadow and return it instead, and the two
    results would still look the same unless something can tell them apart.
    `.as_posix()` matters here — `name.split("/")` does not split backslashes,
    so a bare Windows path would silently skip the fallback branch too and
    let the test pass for the wrong reason.
    """
    explicit = tmp_path / "elsewhere" / "Qwen2.5-1.5B-Instruct"
    explicit.mkdir(parents=True)
    shadow = tmp_path / "models" / "Qwen2.5-1.5B-Instruct"
    shadow.mkdir(parents=True)
    (shadow / "config.json").write_text("{}")

    source = model_source(explicit.as_posix(), models_dir=tmp_path / "models")
    assert source == explicit.as_posix()


def test_a_models_folder_without_config_json_falls_back_to_the_hub_id(tmp_path):
    """`models/<name>` existing is not enough on its own — an empty or
    half-populated directory is not a usable checkpoint, so the result must
    still be the hub id, not that directory."""
    empty = tmp_path / "models" / "Qwen2.5-1.5B-Instruct"
    empty.mkdir(parents=True)

    source = model_source("Qwen/Qwen2.5-1.5B-Instruct", models_dir=tmp_path / "models")
    assert source == "Qwen/Qwen2.5-1.5B-Instruct"
