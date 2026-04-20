"""Tests for ExcerptStore (Stage 6).

All tests use an in-process ephemeral ChromaDB (":memory:") — no filesystem
or Docker required.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from src.ingestion.vlm_extractor import _build_extracted_policy
from src.models.policy import ExtractedPolicy, Payer, Plan, Policy, SourceExcerpt
from src.vector_store.excerpt_store import ExcerptResult, ExcerptStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_store() -> ExcerptStore:
    """Return a fresh isolated ephemeral store.

    Each call uses a unique collection name to prevent cross-test state
    bleed from the in-process EphemeralClient.
    """
    return ExcerptStore(persist_dir=":memory:", collection_name=f"test_{uuid.uuid4().hex}")


def _make_excerpts(policy_id: str = "pol_001", count: int = 3) -> list[SourceExcerpt]:
    texts = [
        "Prior authorization is required for all biologic agents including adalimumab.",
        "Step therapy: patient must fail methotrexate for at least 3 months before initiating biologic therapy.",
        "Quantity limit: two prefilled syringes or autoinjectors per 28-day supply.",
        "Age requirement: patient must be 18 years of age or older.",
        "Diagnosis must be documented by a board-certified rheumatologist.",
    ]
    return [
        SourceExcerpt(
            excerpt_id=f"{policy_id}_exc_{i}",
            policy_id=policy_id,
            text=texts[i % len(texts)],
            page_number=i + 1,
            topic=["prior_auth", "step_therapy", "quantity_limit"][i % 3],
        )
        for i in range(count)
    ]


# ── add_excerpts ──────────────────────────────────────────────────────────────

def test_add_excerpts_returns_count() -> None:
    store = _make_store()
    excerpts = _make_excerpts(count=3)
    assert store.add_excerpts(excerpts) == 3


def test_add_excerpts_empty_list() -> None:
    store = _make_store()
    assert store.add_excerpts([]) == 0


def test_add_excerpts_skips_blank_text() -> None:
    store = _make_store()
    excerpts = [
        SourceExcerpt(excerpt_id="e1", policy_id="p1", text="Real content here.", page_number=1),
        SourceExcerpt(excerpt_id="e2", policy_id="p1", text="   ", page_number=2),
    ]
    assert store.add_excerpts(excerpts) == 1


def test_add_excerpts_idempotent() -> None:
    store = _make_store()
    excerpts = _make_excerpts(count=2)
    store.add_excerpts(excerpts)
    store.add_excerpts(excerpts)          # second upsert of same IDs
    assert store.count() == 2


def test_add_excerpts_stores_bbox() -> None:
    store = _make_store()
    exc = SourceExcerpt(
        excerpt_id="e_bbox",
        policy_id="pol_001",
        text="Prior auth required.",
        page_number=5,
        bbox=[10.0, 20.0, 200.0, 40.0],
    )
    store.add_excerpts([exc])
    results = store.get_by_policy("pol_001")
    assert results[0].bbox == pytest.approx([10.0, 20.0, 200.0, 40.0])


# ── count ─────────────────────────────────────────────────────────────────────

def test_count_total() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 2))
    store.add_excerpts(_make_excerpts("pol_002", 3))
    assert store.count() == 5


def test_count_by_policy() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 2))
    store.add_excerpts(_make_excerpts("pol_002", 3))
    assert store.count("pol_001") == 2
    assert store.count("pol_002") == 3


def test_count_empty_store() -> None:
    assert _make_store().count() == 0


# ── delete_policy ────────────────────────────────────────────────────────────

def test_delete_policy_removes_only_target() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 2))
    store.add_excerpts(_make_excerpts("pol_002", 3))
    deleted = store.delete_policy("pol_001")
    assert deleted == 2
    assert store.count("pol_001") == 0
    assert store.count("pol_002") == 3


def test_delete_nonexistent_policy_returns_zero() -> None:
    store = _make_store()
    assert store.delete_policy("ghost_policy") == 0


# ── get_by_policy ─────────────────────────────────────────────────────────────

def test_get_by_policy_ordered_by_page() -> None:
    store = _make_store()
    # insert out of order
    excerpts = [
        SourceExcerpt(excerpt_id="e3", policy_id="pol_x", text="Page 3 text.", page_number=3),
        SourceExcerpt(excerpt_id="e1", policy_id="pol_x", text="Page 1 text.", page_number=1),
        SourceExcerpt(excerpt_id="e2", policy_id="pol_x", text="Page 2 text.", page_number=2),
    ]
    store.add_excerpts(excerpts)
    results = store.get_by_policy("pol_x")
    assert [r.page_number for r in results] == [1, 2, 3]


def test_get_by_policy_returns_empty_for_unknown() -> None:
    store = _make_store()
    assert store.get_by_policy("unknown_policy") == []


def test_get_by_policy_topic_filter() -> None:
    store = _make_store()
    excerpts = [
        SourceExcerpt(excerpt_id="e1", policy_id="p", text="PA required.", page_number=1, topic="prior_auth"),
        SourceExcerpt(excerpt_id="e2", policy_id="p", text="Step therapy.", page_number=2, topic="step_therapy"),
    ]
    store.add_excerpts(excerpts)
    results = store.get_by_policy("p", topic="prior_auth")
    assert len(results) == 1
    assert results[0].topic == "prior_auth"


# ── search ────────────────────────────────────────────────────────────────────

def test_search_returns_results() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 5))
    results = store.search("prior authorization biologic", n_results=3)
    assert len(results) <= 3
    assert all(isinstance(r, ExcerptResult) for r in results)


def test_search_empty_store_returns_empty() -> None:
    store = _make_store()
    assert store.search("anything") == []


def test_search_policy_filter() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 3))
    store.add_excerpts(_make_excerpts("pol_002", 3))
    results = store.search("step therapy methotrexate", policy_id="pol_001")
    assert all(r.policy_id == "pol_001" for r in results)


def test_search_result_fields_populated() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 2))
    results = store.search("prior authorization", n_results=1)
    assert len(results) == 1
    r = results[0]
    assert r.excerpt_id != ""
    assert r.policy_id == "pol_001"
    assert r.page_number >= 1
    assert 0.0 <= r.relevance <= 1.0
    assert r.distance >= 0.0


def test_search_relevance_inversely_related_to_distance() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 3))
    results = store.search("prior authorization", n_results=3)
    for r in results:
        assert abs(r.relevance - (1.0 - r.distance)) < 1e-6


def test_search_topic_filter() -> None:
    store = _make_store()
    excerpts = [
        SourceExcerpt(excerpt_id="pa1",  policy_id="p", text="PA is required for this drug.",        page_number=1, topic="prior_auth"),
        SourceExcerpt(excerpt_id="st1",  policy_id="p", text="Step therapy trial must be completed.", page_number=2, topic="step_therapy"),
        SourceExcerpt(excerpt_id="ql1",  policy_id="p", text="Quantity limited to 30-day supply.",    page_number=3, topic="quantity_limit"),
    ]
    store.add_excerpts(excerpts)
    results = store.search("authorization requirement", topic="prior_auth", n_results=5)
    assert all(r.topic == "prior_auth" for r in results)


def test_search_n_results_respected() -> None:
    store = _make_store()
    store.add_excerpts(_make_excerpts("pol_001", 5))
    results = store.search("therapy", n_results=2)
    assert len(results) <= 2


# ── ExcerptResult.citation ────────────────────────────────────────────────────

def test_excerpt_result_citation_format() -> None:
    r = ExcerptResult(
        excerpt_id="e1", policy_id="pol_001", text="some text",
        page_number=7, topic="prior_auth", distance=0.1, relevance=0.9,
    )
    assert r.citation == "Policy pol_001, page 7"


# ── add_from_policy ───────────────────────────────────────────────────────────

def test_add_from_policy_empty_excerpts() -> None:
    store = _make_store()
    raw = json.loads((FIXTURES_DIR / "sample_extraction_response.json").read_text())
    ep = _build_extracted_policy(raw, model="test")
    # fixture has no excerpts — should silently return 0
    assert store.add_from_policy(ep) == 0


def test_add_from_policy_with_excerpts() -> None:
    store = _make_store()
    raw = json.loads((FIXTURES_DIR / "sample_extraction_response.json").read_text())
    ep = _build_extracted_policy(raw, model="test")
    # Inject excerpts directly
    ep.excerpts.extend(_make_excerpts(ep.policy.policy_id, 3))
    added = store.add_from_policy(ep)
    assert added == 3
    assert store.count(ep.policy.policy_id) == 3


# ── _build_where coverage ─────────────────────────────────────────────────────

def test_build_where_none_when_no_filters() -> None:
    from src.vector_store.excerpt_store import ExcerptStore as ES
    assert ES._build_where(None, None) is None


def test_build_where_single_policy() -> None:
    from src.vector_store.excerpt_store import ExcerptStore as ES
    w = ES._build_where("pol_001", None)
    assert w == {"policy_id": {"$eq": "pol_001"}}


def test_build_where_both_filters() -> None:
    from src.vector_store.excerpt_store import ExcerptStore as ES
    w = ES._build_where("pol_001", "prior_auth")
    assert w is not None
    assert "$and" in w
    assert len(w["$and"]) == 2
