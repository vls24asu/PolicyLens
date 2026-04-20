"""Tests for Neo4j graph schema and query helpers (Stage 3).

Unit tests (no live database) run always.
Integration tests are skipped unless NEO4J_URI is reachable.
"""

from __future__ import annotations

import os
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from src.graph.schema import (
    UNIQUENESS_CONSTRAINTS,
    LOOKUP_INDEXES,
    FULLTEXT_INDEXES,
    get_schema_cypher,
)
from src.graph import queries


# ── Schema DDL unit tests ──────────────────────────────────────────────────────

def test_schema_covers_all_node_labels() -> None:
    constrained = {label for label, _ in UNIQUENESS_CONSTRAINTS}
    expected = {"Payer", "Plan", "Policy", "Drug", "Indication", "Criterion", "SourceExcerpt"}
    assert expected == constrained


def test_schema_cypher_uses_if_not_exists() -> None:
    stmts = get_schema_cypher()
    for stmt in stmts:
        assert "IF NOT EXISTS" in stmt, f"Missing IF NOT EXISTS: {stmt}"


def test_schema_cypher_count() -> None:
    stmts = get_schema_cypher()
    expected = len(UNIQUENESS_CONSTRAINTS) + len(LOOKUP_INDEXES) + len(FULLTEXT_INDEXES)
    assert len(stmts) == expected


def test_schema_no_string_interpolation_of_user_input() -> None:
    """Schema DDL uses only hardcoded labels / properties — no $params."""
    stmts = get_schema_cypher()
    for stmt in stmts:
        assert "$" not in stmt, f"Unexpected param placeholder in schema DDL: {stmt}"


def test_lookup_indexes_are_on_known_labels() -> None:
    known_labels = {label for label, _ in UNIQUENESS_CONSTRAINTS}
    for label, _ in LOOKUP_INDEXES:
        assert label in known_labels, f"Unknown label in LOOKUP_INDEXES: {label}"


# ── Client unit tests (mocked driver) ─────────────────────────────────────────

@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    client.run = AsyncMock(return_value=[])
    client.run_write = AsyncMock(return_value=[])
    client.run_write_many = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_upsert_payer_calls_run_write(mock_client: MagicMock) -> None:
    await queries.upsert_payer(mock_client, {"payer_id": "aetna", "name": "Aetna", "type": "commercial"})
    mock_client.run_write.assert_awaited_once()
    call_args = mock_client.run_write.call_args
    assert "payer_id" in call_args.args[1]
    assert call_args.args[1]["payer_id"] == "aetna"


@pytest.mark.asyncio
async def test_upsert_drug_calls_run_write(mock_client: MagicMock) -> None:
    await queries.upsert_drug(mock_client, {"drug_id": "adalimumab", "name": "adalimumab"})
    mock_client.run_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_coverage_no_indication(mock_client: MagicMock) -> None:
    mock_client.run.return_value = [
        {"payer": "Aetna", "plan": "Aetna PPO", "coverage_status": "covered", "tier": "Tier 3"}
    ]
    results = await queries.query_coverage(mock_client, "adalimumab")
    mock_client.run.assert_awaited_once()
    cypher: str = mock_client.run.call_args.args[0]
    assert "$drug_name" in cypher
    assert "$indication_name" not in cypher
    assert len(results) == 1


@pytest.mark.asyncio
async def test_query_coverage_with_indication(mock_client: MagicMock) -> None:
    mock_client.run.return_value = []
    await queries.query_coverage(mock_client, "pembrolizumab", indication_name="NSCLC")
    cypher: str = mock_client.run.call_args.args[0]
    assert "$indication_name" in cypher


@pytest.mark.asyncio
async def test_compare_policies_structure(mock_client: MagicMock) -> None:
    mock_client.run.return_value = []
    result = await queries.compare_policies(mock_client, "adalimumab", "Aetna PPO", "Cigna PPO")
    assert result["drug"] == "adalimumab"
    assert "Aetna PPO" in result
    assert "Cigna PPO" in result


@pytest.mark.asyncio
async def test_list_payers_no_params(mock_client: MagicMock) -> None:
    mock_client.run.return_value = [{"payer_id": "aetna", "name": "Aetna", "plan_count": 2}]
    result = await queries.list_payers(mock_client)
    mock_client.run.assert_awaited_once()
    # list_payers passes no params dict — verify call signature
    call_args = mock_client.run.call_args
    assert len(call_args.args) == 1  # only the cypher string


@pytest.mark.asyncio
async def test_get_recent_policy_changes_builds_filter(mock_client: MagicMock) -> None:
    mock_client.run.return_value = []
    await queries.get_recent_policy_changes(
        mock_client, payer_id="aetna", start_date="2024-01-01", end_date="2024-12-31"
    )
    cypher: str = mock_client.run.call_args.args[0]
    params: dict[str, Any] = mock_client.run.call_args.args[1]
    assert "$payer_id" in cypher
    assert "$start_date" in cypher
    assert "$end_date" in cypher
    assert params["payer_id"] == "aetna"


@pytest.mark.asyncio
async def test_get_recent_policy_changes_no_filter(mock_client: MagicMock) -> None:
    mock_client.run.return_value = []
    await queries.get_recent_policy_changes(mock_client)
    cypher: str = mock_client.run.call_args.args[0]
    assert "WHERE" not in cypher


@pytest.mark.asyncio
async def test_link_payer_plan_passes_correct_params(mock_client: MagicMock) -> None:
    await queries.link_payer_plan(mock_client, "aetna", "aetna_ppo_2024")
    params = mock_client.run_write.call_args.args[1]
    assert params["payer_id"] == "aetna"
    assert params["plan_id"] == "aetna_ppo_2024"


@pytest.mark.asyncio
async def test_link_policy_covers_drug_with_tier(mock_client: MagicMock) -> None:
    await queries.link_policy_covers_drug(
        mock_client, "pol_001", "adalimumab", "covered", tier="Tier 3"
    )
    params = mock_client.run_write.call_args.args[1]
    assert params["tier"] == "Tier 3"
    assert params["coverage_status"] == "covered"


@pytest.mark.asyncio
async def test_list_policies_filtered_by_payer(mock_client: MagicMock) -> None:
    mock_client.run.return_value = []
    await queries.list_policies(mock_client, payer_id="cigna")
    params = mock_client.run.call_args.args[1]
    assert params["payer_id"] == "cigna"


@pytest.mark.asyncio
async def test_get_policy_details_returns_none_when_empty(mock_client: MagicMock) -> None:
    mock_client.run.return_value = []
    result = await queries.get_policy_details(mock_client, "nonexistent")
    assert result is None


# ── Integration tests (skipped without live Neo4j) ────────────────────────────

NEO4J_AVAILABLE = bool(os.getenv("NEO4J_URI"))

@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="No live Neo4j instance")
@pytest.mark.asyncio
async def test_integration_schema_init() -> None:
    from src.graph.client import Neo4jClient
    from src.graph.schema import init_schema
    async with Neo4jClient.from_settings() as client:
        await init_schema(client)  # must not raise
        result = await client.run("RETURN 1 AS n")
        assert result[0]["n"] == 1
