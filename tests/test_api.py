"""Tests for the FastAPI backend (Stage 9).

Uses FastAPI's TestClient (synchronous) to exercise all endpoints with
mocked Neo4j, ExcerptStore, and Anthropic clients.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture()
def client() -> TestClient:
    """TestClient with all external dependencies mocked."""
    import api.main as main_mod

    # Reset lazy singletons
    main_mod._neo4j_client  = None
    main_mod._excerpt_store = None
    main_mod._anthropic_client = None

    from fastapi.testclient import TestClient
    return TestClient(main_mod.app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── /policies ─────────────────────────────────────────────────────────────────

def test_list_policies_empty(client: TestClient) -> None:
    with patch("src.graph.queries.list_policies", AsyncMock(return_value=[])):
        with patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())):
            r = client.get("/policies")
    assert r.status_code == 200
    assert r.json() == []


def test_list_policies_returns_rows(client: TestClient) -> None:
    rows = [{"policy_id": "pol_001", "title": "PA Policy", "payer": "Aetna"}]
    with patch("src.graph.queries.list_policies", AsyncMock(return_value=rows)):
        with patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())):
            r = client.get("/policies")
    assert r.status_code == 200
    assert r.json()[0]["policy_id"] == "pol_001"


def test_list_policies_payer_filter_passed(client: TestClient) -> None:
    with patch("src.graph.queries.list_policies", AsyncMock(return_value=[])) as mock_fn:
        with patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())):
            client.get("/policies?payer_id=aetna")
    mock_fn.assert_awaited_once()
    assert mock_fn.call_args.kwargs["payer_id"] == "aetna"


# ── /policies/{policy_id} ─────────────────────────────────────────────────────

def test_get_policy_found(client: TestClient) -> None:
    detail = {"policy_id": "pol_001", "title": "PA Policy", "drugs": []}
    with patch("src.graph.queries.get_policy_details", AsyncMock(return_value=detail)):
        with patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())):
            r = client.get("/policies/pol_001")
    assert r.status_code == 200
    assert r.json()["policy_id"] == "pol_001"


def test_get_policy_not_found(client: TestClient) -> None:
    with patch("src.graph.queries.get_policy_details", AsyncMock(return_value=None)):
        with patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())):
            r = client.get("/policies/nonexistent_policy")
    assert r.status_code == 404


# ── /ingest ───────────────────────────────────────────────────────────────────

def _make_tiny_pdf() -> bytes:
    """Return a minimal valid single-page PDF bytes."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Adalimumab Policy\nAetna")
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    except ImportError:
        pytest.skip("PyMuPDF not installed")


@pytest.fixture()
def mock_extraction():
    """Patch the full ingestion pipeline."""
    raw = json.loads((FIXTURES_DIR / "sample_extraction_response.json").read_text())
    from src.ingestion.vlm_extractor import _build_extracted_policy
    extracted = _build_extracted_policy(raw, model="test")

    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = extracted

    mock_builder = MagicMock()
    mock_builder.build = AsyncMock()
    mock_builder.mark_supersedes = AsyncMock()

    mock_store = MagicMock()
    mock_store.add_from_policy.return_value = 0

    return mock_extractor, mock_builder, mock_store, extracted


def test_ingest_success(client: TestClient, mock_extraction: Any) -> None:
    mock_extractor, mock_builder, mock_store, extracted = mock_extraction

    with (
        patch("src.ingestion.vlm_extractor.VLMExtractor", return_value=mock_extractor),
        patch("src.ingestion.graph_builder.GraphBuilder", return_value=mock_builder),
        patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())),
        patch("api.main._get_store", return_value=mock_store),
        patch("api.main._get_anthropic", return_value=MagicMock()),
    ):
        pdf_bytes = _make_tiny_pdf()
        r = client.post(
            "/ingest",
            files={"file": ("test_policy.pdf", pdf_bytes, "application/pdf")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["policy_id"] == extracted.policy.policy_id
    assert body["drugs_found"] == len(extracted.drugs)
    assert body["criteria_found"] == len(extracted.criteria)


def test_ingest_non_pdf_rejected(client: TestClient) -> None:
    r = client.post(
        "/ingest",
        files={"file": ("policy.txt", b"some text", "text/plain")},
    )
    assert r.status_code == 400


def test_ingest_with_supersedes(client: TestClient, mock_extraction: Any) -> None:
    mock_extractor, mock_builder, mock_store, extracted = mock_extraction

    with (
        patch("src.ingestion.vlm_extractor.VLMExtractor", return_value=mock_extractor),
        patch("src.ingestion.graph_builder.GraphBuilder", return_value=mock_builder),
        patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())),
        patch("api.main._get_store", return_value=mock_store),
        patch("api.main._get_anthropic", return_value=MagicMock()),
    ):
        pdf_bytes = _make_tiny_pdf()
        r = client.post(
            "/ingest",
            files={"file": ("policy.pdf", pdf_bytes, "application/pdf")},
            data={"supersedes": "old_policy_id"},
        )

    assert r.status_code == 200
    mock_builder.mark_supersedes.assert_awaited_once_with(
        extracted.policy.policy_id, "old_policy_id"
    )


# ── /query ────────────────────────────────────────────────────────────────────

def _make_mock_anthropic(tool_calls: list[dict], final_text: str) -> MagicMock:
    """Build a mock Anthropic client that returns tool calls then a final answer."""
    responses = []

    # One response per set of tool calls
    for tc_batch in tool_calls:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id   = "tool_use_abc"
        tool_block.name = tc_batch["name"]
        tool_block.input = tc_batch["input"]

        resp = MagicMock()
        resp.stop_reason = "tool_use"
        resp.content = [tool_block]
        resp.model = "claude-sonnet-4-6"
        responses.append(resp)

    # Final text response
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = final_text

    final_resp = MagicMock()
    final_resp.stop_reason = "end_turn"
    final_resp.content = [text_block]
    final_resp.model = "claude-sonnet-4-6"
    responses.append(final_resp)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = responses
    return mock_client


def test_query_no_tools(client: TestClient) -> None:
    """Claude answers directly without calling any tools."""
    mock_ac = _make_mock_anthropic([], "Adalimumab is covered by Aetna.")

    with (
        patch("api.main._get_anthropic", return_value=mock_ac),
        patch("api.main._prime_mcp_clients", AsyncMock()),
        patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())),
        patch("api.main._get_store", return_value=MagicMock()),
    ):
        r = client.post("/query", json={"question": "Does Aetna cover adalimumab?"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Adalimumab is covered by Aetna."
    assert body["tool_calls"] == []
    assert body["model"] == "claude-sonnet-4-6"


def test_query_with_tool_call(client: TestClient) -> None:
    """Claude calls one tool then gives a final answer."""
    mock_ac = _make_mock_anthropic(
        [{"name": "list_payers", "input": {}}],
        "Aetna and Cigna are available.",
    )

    mock_tool_result = [{"payer_id": "aetna", "name": "Aetna"}]

    with (
        patch("api.main._get_anthropic", return_value=mock_ac),
        patch("api.main._prime_mcp_clients", AsyncMock()),
        patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())),
        patch("api.main._get_store", return_value=MagicMock()),
        patch("api.main._execute_tool", AsyncMock(return_value=mock_tool_result)),
    ):
        r = client.post("/query", json={"question": "Which payers are available?"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Aetna and Cigna are available."
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["tool_name"] == "list_payers"


def test_query_tool_call_trace_contains_result(client: TestClient) -> None:
    """Tool call log captures both input and result."""
    mock_ac = _make_mock_anthropic(
        [{"name": "query_coverage", "input": {"drug_name": "adalimumab"}}],
        "Covered by Aetna.",
    )
    tool_result = [{"payer": "Aetna", "coverage_status": "covered"}]

    with (
        patch("api.main._get_anthropic", return_value=mock_ac),
        patch("api.main._prime_mcp_clients", AsyncMock()),
        patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())),
        patch("api.main._get_store", return_value=MagicMock()),
        patch("api.main._execute_tool", AsyncMock(return_value=tool_result)),
    ):
        r = client.post("/query", json={"question": "Coverage for adalimumab?"})

    body = r.json()
    tc = body["tool_calls"][0]
    assert tc["tool_input"]["drug_name"] == "adalimumab"
    assert tc["tool_result"] == tool_result


def test_query_exceeds_rounds_returns_500(client: TestClient) -> None:
    """If Claude never stops calling tools, the API returns 500."""
    # Always return tool_use stop reason
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tu1"
    tool_block.name = "list_payers"
    tool_block.input = {}

    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [tool_block]
    resp.model = "claude-sonnet-4-6"

    mock_ac = MagicMock()
    mock_ac.messages.create.return_value = resp

    with (
        patch("api.main._get_anthropic", return_value=mock_ac),
        patch("api.main._prime_mcp_clients", AsyncMock()),
        patch("api.main._get_neo4j", AsyncMock(return_value=MagicMock())),
        patch("api.main._get_store", return_value=MagicMock()),
        patch("api.main._execute_tool", AsyncMock(return_value=[])),
    ):
        r = client.post(
            "/query",
            json={"question": "loop forever", "max_tool_rounds": 2},
        )

    assert r.status_code == 500


# ── _build_tool_registry / _build_anthropic_tools ────────────────────────────

def test_tool_registry_contains_all_tools() -> None:
    from api.main import _build_tool_registry
    registry = _build_tool_registry()
    expected = {
        "query_coverage", "get_prior_auth_criteria", "compare_policies",
        "search_policies_by_drug_class", "list_payers",
        "get_policy_excerpt", "get_policy_metadata", "list_policies_for_drug",
        "normalize_drug_name", "lookup_icd10", "get_drug_class",
        "get_policy_changes", "compare_policy_versions", "get_recent_updates",
    }
    missing = expected - set(registry.keys())
    assert not missing, f"Missing tools: {missing}"


def test_anthropic_tools_schema_has_required_fields() -> None:
    from api.main import _build_tool_registry, _build_anthropic_tools
    registry = _build_tool_registry()
    tools = _build_anthropic_tools(registry)
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t
        assert t["input_schema"].get("type") == "object"


# ── _serialise helper ─────────────────────────────────────────────────────────

def test_serialise_dict() -> None:
    from api.main import _serialise
    assert _serialise({"key": "value"}) == '{"key": "value"}'


def test_serialise_non_serialisable_falls_back_to_str() -> None:
    from api.main import _serialise
    class Unserializable:
        def __repr__(self) -> str:
            return "Unserializable()"
    result = _serialise(Unserializable())
    assert isinstance(result, str)
