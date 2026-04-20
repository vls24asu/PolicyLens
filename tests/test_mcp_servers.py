"""Tests for all four MCP servers (Stage 8).

Calls each tool function directly (bypassing the MCP transport layer) with
mocked backing clients.  No live Neo4j, ChromaDB, or network required.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _async_mock(return_value: Any) -> AsyncMock:
    m = AsyncMock()
    m.return_value = return_value
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Policy Graph Server
# ══════════════════════════════════════════════════════════════════════════════

class TestPolicyGraphServer:

    @pytest.fixture(autouse=True)
    def _reset_client(self):
        """Reset module-level lazy client between tests."""
        import src.mcp_servers.policy_graph_server as mod
        mod._neo4j_client = None
        yield
        mod._neo4j_client = None

    @pytest.fixture()
    def mock_neo4j(self):
        from src.mcp_servers import policy_graph_server as mod
        client = MagicMock()
        mod._neo4j_client = client
        return client

    # ── query_coverage ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_query_coverage_returns_results(self, mock_neo4j):
        rows = [{"payer": "Aetna", "plan": "Aetna PPO", "coverage_status": "covered", "tier": "Tier 3"}]
        with patch("src.graph.queries.query_coverage", _async_mock(rows)):
            from src.mcp_servers.policy_graph_server import query_coverage
            result = await query_coverage("adalimumab")
        assert result == rows

    @pytest.mark.asyncio
    async def test_query_coverage_no_results_returns_message(self, mock_neo4j):
        with patch("src.graph.queries.query_coverage", _async_mock([])):
            from src.mcp_servers.policy_graph_server import query_coverage
            result = await query_coverage("unknowndrug99")
        assert len(result) == 1
        assert "message" in result[0]

    @pytest.mark.asyncio
    async def test_query_coverage_passes_indication(self, mock_neo4j):
        with patch("src.graph.queries.query_coverage", _async_mock([])) as mock_fn:
            from src.mcp_servers.policy_graph_server import query_coverage
            await query_coverage("adalimumab", indication="rheumatoid arthritis")
        mock_fn.assert_awaited_once()
        _, kwargs = mock_fn.call_args
        assert kwargs.get("indication_name") == "rheumatoid arthritis"

    @pytest.mark.asyncio
    async def test_query_coverage_empty_indication_passes_none(self, mock_neo4j):
        with patch("src.graph.queries.query_coverage", _async_mock([])) as mock_fn:
            from src.mcp_servers.policy_graph_server import query_coverage
            await query_coverage("adalimumab", indication="  ")
        _, kwargs = mock_fn.call_args
        assert kwargs.get("indication_name") is None

    # ── get_prior_auth_criteria ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_prior_auth_criteria_returns_rows(self, mock_neo4j):
        rows = [{"type": "prior_auth", "description": "PA required"}]
        with patch("src.graph.queries.get_prior_auth_criteria", _async_mock(rows)):
            from src.mcp_servers.policy_graph_server import get_prior_auth_criteria
            result = await get_prior_auth_criteria("adalimumab", "Aetna PPO")
        assert result == rows

    @pytest.mark.asyncio
    async def test_get_prior_auth_criteria_no_match_message(self, mock_neo4j):
        with patch("src.graph.queries.get_prior_auth_criteria", _async_mock([])):
            from src.mcp_servers.policy_graph_server import get_prior_auth_criteria
            result = await get_prior_auth_criteria("unknowndrug", "Unknown Plan")
        assert "message" in result[0]

    # ── compare_policies ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_compare_policies_returns_dict(self, mock_neo4j):
        comparison = {"drug": "adalimumab", "Aetna PPO": {"tier": "Tier 3"}, "Cigna PPO": None}
        with patch("src.graph.queries.compare_policies", _async_mock(comparison)):
            from src.mcp_servers.policy_graph_server import compare_policies
            result = await compare_policies("adalimumab", "Aetna PPO", "Cigna PPO")
        assert result["drug"] == "adalimumab"

    # ── search_policies_by_drug_class ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_search_by_drug_class_returns_list(self, mock_neo4j):
        rows = [{"policy_id": "pol_001", "title": "Biologic PA", "drugs": ["adalimumab"]}]
        with patch("src.graph.queries.search_policies_by_drug_class", _async_mock(rows)):
            from src.mcp_servers.policy_graph_server import search_policies_by_drug_class
            result = await search_policies_by_drug_class("TNF inhibitor")
        assert result == rows

    # ── list_payers ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_list_payers_returns_list(self, mock_neo4j):
        payers = [{"payer_id": "aetna", "name": "Aetna", "plan_count": 2}]
        with patch("src.graph.queries.list_payers", _async_mock(payers)):
            from src.mcp_servers.policy_graph_server import list_payers
            result = await list_payers()
        assert result == payers

    @pytest.mark.asyncio
    async def test_list_payers_empty_db_message(self, mock_neo4j):
        with patch("src.graph.queries.list_payers", _async_mock([])):
            from src.mcp_servers.policy_graph_server import list_payers
            result = await list_payers()
        assert "message" in result[0]


# ══════════════════════════════════════════════════════════════════════════════
# Document Retrieval Server
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentRetrievalServer:

    @pytest.fixture(autouse=True)
    def _reset(self):
        import src.mcp_servers.document_retrieval_server as mod
        mod._neo4j_client = None
        mod._excerpt_store = None
        yield
        mod._neo4j_client = None
        mod._excerpt_store = None

    # ── get_policy_excerpt ────────────────────────────────────────────────────

    def test_get_policy_excerpt_returns_formatted(self):
        from src.vector_store.excerpt_store import ExcerptResult
        mock_result = ExcerptResult(
            excerpt_id="exc_01", policy_id="pol_001",
            text="PA is required.", page_number=3,
            topic="prior_auth", distance=0.1, relevance=0.9,
        )
        mock_store = MagicMock()
        mock_store.search.return_value = [mock_result]

        import src.mcp_servers.document_retrieval_server as mod
        mod._excerpt_store = mock_store

        from src.mcp_servers.document_retrieval_server import get_policy_excerpt
        result = get_policy_excerpt("pol_001", "prior authorization", n_results=3)
        assert len(result) == 1
        assert result[0]["text"] == "PA is required."
        assert result[0]["page_number"] == 3
        assert result[0]["relevance"] == pytest.approx(0.9, abs=0.001)
        assert "citation" in result[0]

    def test_get_policy_excerpt_no_results_message(self):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        import src.mcp_servers.document_retrieval_server as mod
        mod._excerpt_store = mock_store

        from src.mcp_servers.document_retrieval_server import get_policy_excerpt
        result = get_policy_excerpt("pol_999", "anything")
        assert "message" in result[0]

    def test_get_policy_excerpt_passes_n_results(self):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        import src.mcp_servers.document_retrieval_server as mod
        mod._excerpt_store = mock_store

        from src.mcp_servers.document_retrieval_server import get_policy_excerpt
        get_policy_excerpt("pol_001", "step therapy", n_results=7)
        mock_store.search.assert_called_once_with("step therapy", policy_id="pol_001", n_results=7)

    # ── get_policy_metadata ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_policy_metadata_returns_detail(self):
        detail = {"policy_id": "pol_001", "title": "Biologic PA", "payer": "Aetna"}
        import src.mcp_servers.document_retrieval_server as mod
        mod._neo4j_client = MagicMock()

        with patch("src.graph.queries.get_policy_details", _async_mock(detail)):
            from src.mcp_servers.document_retrieval_server import get_policy_metadata
            result = await get_policy_metadata("pol_001")
        assert result["policy_id"] == "pol_001"

    @pytest.mark.asyncio
    async def test_get_policy_metadata_not_found(self):
        import src.mcp_servers.document_retrieval_server as mod
        mod._neo4j_client = MagicMock()

        with patch("src.graph.queries.get_policy_details", _async_mock(None)):
            from src.mcp_servers.document_retrieval_server import get_policy_metadata
            result = await get_policy_metadata("nonexistent")
        assert "error" in result

    # ── list_policies_for_drug ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_list_policies_for_drug_returns_list(self):
        rows = [{"policy_id": "pol_001", "title": "PA Policy", "relationship": "COVERS"}]
        import src.mcp_servers.document_retrieval_server as mod
        mod._neo4j_client = MagicMock()

        with patch("src.graph.queries.list_policies_for_drug", _async_mock(rows)):
            from src.mcp_servers.document_retrieval_server import list_policies_for_drug
            result = await list_policies_for_drug("adalimumab")
        assert result == rows

    @pytest.mark.asyncio
    async def test_list_policies_for_drug_no_results(self):
        import src.mcp_servers.document_retrieval_server as mod
        mod._neo4j_client = MagicMock()

        with patch("src.graph.queries.list_policies_for_drug", _async_mock([])):
            from src.mcp_servers.document_retrieval_server import list_policies_for_drug
            result = await list_policies_for_drug("xyzunknown")
        assert "message" in result[0]


# ══════════════════════════════════════════════════════════════════════════════
# Reference Data Server
# ══════════════════════════════════════════════════════════════════════════════

class TestReferenceDataServer:

    @pytest.fixture(autouse=True)
    def _reset(self):
        import src.mcp_servers.reference_data_server as mod
        mod._rxnorm_client = None
        mod._icd10_client = None
        yield
        mod._rxnorm_client = None
        mod._icd10_client = None

    # ── normalize_drug_name ───────────────────────────────────────────────────

    def test_normalize_drug_name_found(self):
        from src.reference.rxnorm import RxNormResult
        mock_rxnorm = MagicMock()
        mock_rxnorm.normalize.return_value = RxNormResult(
            rxnorm_cui="327361", standard_name="adalimumab",
            brand_names=["Humira"], drug_classes=["TNF inhibitor"],
        )
        import src.mcp_servers.reference_data_server as mod
        mod._rxnorm_client = mock_rxnorm

        from src.mcp_servers.reference_data_server import normalize_drug_name
        result = normalize_drug_name("Humira")
        assert result["rxnorm_cui"] == "327361"
        assert result["standard_name"] == "adalimumab"
        assert "Humira" in result["brand_names"]

    def test_normalize_drug_name_not_found(self):
        mock_rxnorm = MagicMock()
        mock_rxnorm.normalize.return_value = None
        import src.mcp_servers.reference_data_server as mod
        mod._rxnorm_client = mock_rxnorm

        from src.mcp_servers.reference_data_server import normalize_drug_name
        result = normalize_drug_name("xyzunknown")
        assert "error" in result

    def test_normalize_drug_name_synonyms_trimmed(self):
        from src.reference.rxnorm import RxNormResult
        mock_rxnorm = MagicMock()
        mock_rxnorm.normalize.return_value = RxNormResult(
            rxnorm_cui="1", standard_name="test",
            synonyms=["a", "b", "c", "d", "e", "f", "g"],  # 7 synonyms
        )
        import src.mcp_servers.reference_data_server as mod
        mod._rxnorm_client = mock_rxnorm

        from src.mcp_servers.reference_data_server import normalize_drug_name
        result = normalize_drug_name("test")
        assert len(result["synonyms"]) <= 5

    # ── lookup_icd10 ──────────────────────────────────────────────────────────

    def test_lookup_icd10_returns_codes(self):
        from src.reference.icd10 import ICD10Result
        mock_icd10 = MagicMock()
        mock_icd10.search.return_value = [
            ICD10Result(code="M05.79", description="RA multiple sites"),
        ]
        import src.mcp_servers.reference_data_server as mod
        mod._icd10_client = mock_icd10

        from src.mcp_servers.reference_data_server import lookup_icd10
        result = lookup_icd10("rheumatoid arthritis")
        assert result[0]["code"] == "M05.79"

    def test_lookup_icd10_no_results_error(self):
        mock_icd10 = MagicMock()
        mock_icd10.search.return_value = []
        import src.mcp_servers.reference_data_server as mod
        mod._icd10_client = mock_icd10

        from src.mcp_servers.reference_data_server import lookup_icd10
        result = lookup_icd10("xyznonexistent")
        assert "error" in result[0]

    def test_lookup_icd10_passes_max_results(self):
        mock_icd10 = MagicMock()
        mock_icd10.search.return_value = []
        import src.mcp_servers.reference_data_server as mod
        mod._icd10_client = mock_icd10

        from src.mcp_servers.reference_data_server import lookup_icd10
        lookup_icd10("psoriasis", max_results=15)
        mock_icd10.search.assert_called_once_with("psoriasis", max_results=15)

    # ── get_drug_class ────────────────────────────────────────────────────────

    def test_get_drug_class_returns_classes(self):
        from src.reference.rxnorm import DrugClassResult, RxNormResult
        mock_rxnorm = MagicMock()
        mock_rxnorm.normalize.return_value = RxNormResult(
            rxnorm_cui="327361", standard_name="adalimumab"
        )
        mock_rxnorm.get_drug_classes.return_value = [
            DrugClassResult(class_id="N01", class_name="TNF inhibitor", class_type="EPC"),
        ]
        import src.mcp_servers.reference_data_server as mod
        mod._rxnorm_client = mock_rxnorm

        from src.mcp_servers.reference_data_server import get_drug_class
        result = get_drug_class("adalimumab")
        assert result["rxnorm_cui"] == "327361"
        assert result["classes"][0]["class_name"] == "TNF inhibitor"

    def test_get_drug_class_not_found(self):
        mock_rxnorm = MagicMock()
        mock_rxnorm.normalize.return_value = None
        import src.mcp_servers.reference_data_server as mod
        mod._rxnorm_client = mock_rxnorm

        from src.mcp_servers.reference_data_server import get_drug_class
        result = get_drug_class("unknown")
        assert "error" in result


# ══════════════════════════════════════════════════════════════════════════════
# Change Detection Server
# ══════════════════════════════════════════════════════════════════════════════

class TestChangeDetectionServer:

    @pytest.fixture(autouse=True)
    def _reset(self):
        import src.mcp_servers.change_detection_server as mod
        mod._neo4j_client = None
        yield
        mod._neo4j_client = None

    @pytest.fixture()
    def mock_neo4j(self):
        import src.mcp_servers.change_detection_server as mod
        client = MagicMock()
        mod._neo4j_client = client
        return client

    # ── get_policy_changes ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_policy_changes_returns_rows(self, mock_neo4j):
        rows = [{"policy_id": "pol_001", "effective_date": "2024-06-01", "payer": "Aetna"}]
        with patch("src.graph.queries.get_recent_policy_changes", _async_mock(rows)):
            from src.mcp_servers.change_detection_server import get_policy_changes
            result = await get_policy_changes(start_date="2024-01-01", end_date="2024-12-31")
        assert result == rows

    @pytest.mark.asyncio
    async def test_get_policy_changes_empty_strings_pass_none(self, mock_neo4j):
        with patch("src.graph.queries.get_recent_policy_changes", _async_mock([])) as mock_fn:
            from src.mcp_servers.change_detection_server import get_policy_changes
            await get_policy_changes(payer="", start_date="", end_date="")
        _, kwargs = mock_fn.call_args
        assert kwargs["payer_id"] is None
        assert kwargs["start_date"] is None
        assert kwargs["end_date"] is None

    @pytest.mark.asyncio
    async def test_get_policy_changes_no_results_message(self, mock_neo4j):
        with patch("src.graph.queries.get_recent_policy_changes", _async_mock([])):
            from src.mcp_servers.change_detection_server import get_policy_changes
            result = await get_policy_changes()
        assert "message" in result[0]

    # ── compare_policy_versions ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_compare_versions_adds_and_removes(self, mock_neo4j):
        old = {
            "policy_id": "pol_v1", "title": "PA v1", "version": "1.0",
            "effective_date": "2023-01-01", "last_reviewed_date": None,
            "drugs": [{"drug_id": "adalimumab"}, {"drug_id": "methotrexate"}],
            "criteria": [{"criterion_id": "c1"}],
        }
        new = {
            "policy_id": "pol_v2", "title": "PA v2", "version": "2.0",
            "effective_date": "2024-01-01", "last_reviewed_date": None,
            "drugs": [{"drug_id": "adalimumab"}, {"drug_id": "etanercept"}],
            "criteria": [{"criterion_id": "c1"}, {"criterion_id": "c2"}],
        }
        mock_fn = AsyncMock(side_effect=[old, new])
        with patch("src.graph.queries.get_policy_details", mock_fn):
            from src.mcp_servers.change_detection_server import compare_policy_versions
            result = await compare_policy_versions("pol_v1", "pol_v2")

        assert "etanercept" in result["added_drugs"]
        assert "methotrexate" in result["removed_drugs"]
        assert "adalimumab" in result["unchanged_drugs"]
        assert "c2" in result["added_criteria"]
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_compare_versions_old_not_found(self, mock_neo4j):
        mock_fn = AsyncMock(side_effect=[None, {}])
        with patch("src.graph.queries.get_policy_details", mock_fn):
            from src.mcp_servers.change_detection_server import compare_policy_versions
            result = await compare_policy_versions("missing_v1", "pol_v2")
        assert "error" in result

    # ── get_recent_updates ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_recent_updates_filters_by_date(self, mock_neo4j):
        from datetime import date, timedelta
        today = date.today()
        recent_date = (today - timedelta(days=30)).isoformat()
        old_date    = (today - timedelta(days=200)).isoformat()

        all_policies = [
            {"policy_id": "pol_new", "effective_date": recent_date, "payer": "Aetna"},
            {"policy_id": "pol_old", "effective_date": old_date,    "payer": "Cigna"},
        ]
        with patch("src.graph.queries.list_policies_for_drug", _async_mock(all_policies)):
            from src.mcp_servers.change_detection_server import get_recent_updates
            result = await get_recent_updates("adalimumab", days=90)

        ids = [r["policy_id"] for r in result]
        assert "pol_new" in ids
        assert "pol_old" not in ids

    @pytest.mark.asyncio
    async def test_get_recent_updates_no_drug_found(self, mock_neo4j):
        with patch("src.graph.queries.list_policies_for_drug", _async_mock([])):
            from src.mcp_servers.change_detection_server import get_recent_updates
            result = await get_recent_updates("unknowndrug")
        assert "message" in result[0]

    @pytest.mark.asyncio
    async def test_get_recent_updates_all_old_returns_message(self, mock_neo4j):
        from datetime import date, timedelta
        old_date = (date.today() - timedelta(days=500)).isoformat()
        with patch("src.graph.queries.list_policies_for_drug", _async_mock([
            {"policy_id": "pol_ancient", "effective_date": old_date}
        ])):
            from src.mcp_servers.change_detection_server import get_recent_updates
            result = await get_recent_updates("adalimumab", days=90)
        assert "message" in result[0]

    # ── _diff_policies unit tests ─────────────────────────────────────────────

    def test_diff_metadata_changes(self):
        from src.mcp_servers.change_detection_server import _diff_policies
        old = {"policy_id": "v1", "title": "Old Title", "version": "1.0",
               "effective_date": "2023-01-01", "last_reviewed_date": None,
               "drugs": [], "criteria": []}
        new = {"policy_id": "v2", "title": "New Title", "version": "2.0",
               "effective_date": "2024-01-01", "last_reviewed_date": None,
               "drugs": [], "criteria": []}
        result = _diff_policies(old, new)
        fields_changed = [c["field"] for c in result["metadata_changes"]]
        assert "title" in fields_changed
        assert "version" in fields_changed
        assert "effective_date" in fields_changed

    def test_diff_no_changes(self):
        from src.mcp_servers.change_detection_server import _diff_policies
        same = {"policy_id": "v1", "title": "Same", "version": "1.0",
                "effective_date": "2024-01-01", "last_reviewed_date": None,
                "drugs": [{"drug_id": "adalimumab"}],
                "criteria": [{"criterion_id": "c1"}]}
        result = _diff_policies(same, same)
        assert result["added_drugs"] == []
        assert result["removed_drugs"] == []
        assert result["metadata_changes"] == []


# ══════════════════════════════════════════════════════════════════════════════
# Server importability smoke tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("module_path", [
    "src.mcp_servers.policy_graph_server",
    "src.mcp_servers.document_retrieval_server",
    "src.mcp_servers.reference_data_server",
    "src.mcp_servers.change_detection_server",
])
def test_server_module_importable(module_path: str) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "mcp"), f"{module_path} must expose a 'mcp' FastMCP instance"


@pytest.mark.parametrize("module_path,expected_tools", [
    ("src.mcp_servers.policy_graph_server",
     ["query_coverage", "get_prior_auth_criteria", "compare_policies",
      "search_policies_by_drug_class", "list_payers"]),
    ("src.mcp_servers.document_retrieval_server",
     ["get_policy_excerpt", "get_policy_metadata", "list_policies_for_drug"]),
    ("src.mcp_servers.reference_data_server",
     ["normalize_drug_name", "lookup_icd10", "get_drug_class"]),
    ("src.mcp_servers.change_detection_server",
     ["get_policy_changes", "compare_policy_versions", "get_recent_updates"]),
])
def test_server_registers_all_tools(module_path: str, expected_tools: list[str]) -> None:
    mod = importlib.import_module(module_path)
    # FastMCP stores tools in ._tool_manager._tools dict
    registered = set(mod.mcp._tool_manager._tools.keys())
    for tool_name in expected_tools:
        assert tool_name in registered, (
            f"Tool '{tool_name}' not registered in {module_path}. Found: {registered}"
        )
