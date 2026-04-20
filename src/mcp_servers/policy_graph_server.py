"""Policy Graph MCP server.

Exposes Neo4j graph queries as MCP tools so Claude can ask structured
questions about drug coverage across payers and plans.

Run standalone:
    python scripts/run_mcp_server.py policy_graph
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── Lazy client ───────────────────────────────────────────────────────────────

_neo4j_client: Any = None


async def _get_neo4j() -> Any:
    """Return a connected Neo4jClient, creating it on first call."""
    global _neo4j_client
    if _neo4j_client is None:
        from src.graph.client import Neo4jClient
        _neo4j_client = Neo4jClient.from_settings()
        await _neo4j_client.connect()
    return _neo4j_client


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncGenerator[None, None]:
    yield
    if _neo4j_client is not None:
        await _neo4j_client.close()


# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "policy-graph",
    instructions=(
        "Query the medical benefit drug policy knowledge graph. "
        "Use these tools to look up drug coverage, prior auth requirements, "
        "and compare policies across health insurance payers."
    ),
    lifespan=_lifespan,
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def query_coverage(drug_name: str, indication: str = "") -> list[dict[str, Any]]:
    """Find all insurance plans that cover a drug, optionally filtered by indication.

    Args:
        drug_name: Generic or brand name of the drug (e.g. "adalimumab", "Humira").
        indication: Optional clinical indication to filter by (e.g. "rheumatoid arthritis").

    Returns:
        List of {payer, plan, policy_id, policy_title, coverage_status, tier}.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    ind = indication.strip() or None
    results = await queries.query_coverage(neo4j, drug_name, indication_name=ind)
    if not results:
        return [{"message": f"No coverage records found for '{drug_name}'."}]
    return results


@mcp.tool()
async def get_prior_auth_criteria(drug_name: str, plan_name: str) -> list[dict[str, Any]]:
    """Return all prior-authorization and step-therapy criteria for a drug under a specific plan.

    Args:
        drug_name: Generic or brand name of the drug.
        plan_name: Exact name of the insurance plan (e.g. "Aetna Commercial PPO").

    Returns:
        List of {criterion_id, type, description, required_value, sequence, policy_id}.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    results = await queries.get_prior_auth_criteria(neo4j, drug_name, plan_name)
    if not results:
        return [{"message": f"No criteria found for '{drug_name}' under '{plan_name}'."}]
    return results


@mcp.tool()
async def compare_policies(drug_name: str, plan_a: str, plan_b: str) -> dict[str, Any]:
    """Compare coverage and requirements for a drug between two insurance plans.

    Args:
        drug_name: Generic or brand name of the drug.
        plan_a: Name of the first plan.
        plan_b: Name of the second plan.

    Returns:
        Dict with keys 'drug', plan_a name, plan_b name — each plan value contains
        coverage_status, tier, effective_date, and a list of criteria.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    return await queries.compare_policies(neo4j, drug_name, plan_a, plan_b)


@mcp.tool()
async def search_policies_by_drug_class(drug_class: str) -> list[dict[str, Any]]:
    """Find all policies that cover any drug in a given therapeutic class.

    Args:
        drug_class: Therapeutic class name (e.g. "TNF inhibitor", "GLP-1 agonist", "anti-VEGF").

    Returns:
        List of {policy_id, title, effective_date, payer, drugs}.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    results = await queries.search_policies_by_drug_class(neo4j, drug_class)
    if not results:
        return [{"message": f"No policies found for drug class '{drug_class}'."}]
    return results


@mcp.tool()
async def list_payers() -> list[dict[str, Any]]:
    """List all payers in the database with plan and policy counts.

    Returns:
        List of {payer_id, name, type, plan_count, policy_count}.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    results = await queries.list_payers(neo4j)
    if not results:
        return [{"message": "No payers found. Have you ingested any policies yet?"}]
    return results


if __name__ == "__main__":
    mcp.run()
