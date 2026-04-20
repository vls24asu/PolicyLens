"""Change Detection MCP server.

Detects and summarises policy changes over time by querying the graph
for recently updated policies and diffing policy versions.

Run standalone:
    python scripts/run_mcp_server.py change_detection
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any, AsyncGenerator

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── Lazy client ───────────────────────────────────────────────────────────────

_neo4j_client: Any = None


async def _get_neo4j() -> Any:
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
    "change-detection",
    instructions=(
        "Track policy changes over time. Use these tools to identify which "
        "payers have updated their drug coverage policies in a date range and "
        "to diff specific policy versions."
    ),
    lifespan=_lifespan,
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def get_policy_changes(
    payer: str = "",
    start_date: str = "",
    end_date: str = "",
) -> list[dict[str, Any]]:
    """Return policies whose effective date falls in a date range.

    Args:
        payer: Payer name or payer_id to filter by (leave empty for all payers).
        start_date: ISO 8601 start date, e.g. "2024-01-01" (leave empty for no lower bound).
        end_date: ISO 8601 end date, e.g. "2024-12-31" (leave empty for no upper bound).

    Returns:
        List of {policy_id, title, effective_date, version, payer} ordered by date desc.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    results = await queries.get_recent_policy_changes(
        neo4j,
        payer_id=payer.strip() or None,
        start_date=start_date.strip() or None,
        end_date=end_date.strip() or None,
    )
    if not results:
        return [{"message": "No policy changes found for the specified criteria."}]
    return results


@mcp.tool()
async def compare_policy_versions(
    policy_id_old: str,
    policy_id_new: str,
) -> dict[str, Any]:
    """Produce a structured diff between two versions of a policy.

    Compares drugs covered, criteria, and metadata between an old and new
    policy_id.  Typically the new policy will have a SUPERSEDES relationship
    pointing to the old one.

    Args:
        policy_id_old: The policy_id of the earlier version.
        policy_id_new: The policy_id of the newer version.

    Returns:
        Dict with added_drugs, removed_drugs, changed_criteria, metadata_changes.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()

    old_detail, new_detail = await _fetch_both(queries, neo4j, policy_id_old, policy_id_new)
    if old_detail is None:
        return {"error": f"Policy '{policy_id_old}' not found."}
    if new_detail is None:
        return {"error": f"Policy '{policy_id_new}' not found."}

    return _diff_policies(old_detail, new_detail)


@mcp.tool()
async def get_recent_updates(drug_name: str, days: int = 90) -> list[dict[str, Any]]:
    """Return recent policy changes that affect a specific drug.

    Args:
        drug_name: Generic or brand name of the drug.
        days: Look-back window in days (default 90).

    Returns:
        List of {policy_id, title, effective_date, payer, relationship}
        for policies updated in the look-back window that mention the drug.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()

    cutoff = (date.today() - timedelta(days=days)).isoformat()

    # Get all policies for the drug
    all_policies = await queries.list_policies_for_drug(neo4j, drug_name)
    if not all_policies:
        return [{"message": f"No policies found mentioning '{drug_name}'."}]

    # Filter to those with effective_date >= cutoff
    recent = [
        p for p in all_policies
        if p.get("effective_date") and p["effective_date"] >= cutoff
    ]

    if not recent:
        return [{
            "message": (
                f"No policy updates found for '{drug_name}' in the last {days} days. "
                f"(cutoff: {cutoff})"
            )
        }]
    return recent


# ── Diff helpers ──────────────────────────────────────────────────────────────

async def _fetch_both(
    queries: Any,
    neo4j: Any,
    old_id: str,
    new_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    import asyncio
    old_task = queries.get_policy_details(neo4j, old_id)
    new_task = queries.get_policy_details(neo4j, new_id)
    old_detail, new_detail = await asyncio.gather(old_task, new_task)
    return old_detail, new_detail


def _diff_policies(
    old: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:
    """Compare two policy detail dicts and return a structured diff."""

    def _drug_ids(detail: dict[str, Any]) -> set[str]:
        return {d.get("drug_id", "") for d in (detail.get("drugs") or []) if d.get("drug_id")}

    def _crit_ids(detail: dict[str, Any]) -> set[str]:
        return {c.get("criterion_id", "") for c in (detail.get("criteria") or []) if c.get("criterion_id")}

    old_drugs = _drug_ids(old)
    new_drugs = _drug_ids(new)
    old_crits = _crit_ids(old)
    new_crits = _crit_ids(new)

    # Metadata changes
    meta_changes: list[dict[str, Any]] = []
    for field in ("title", "version", "effective_date", "last_reviewed_date"):
        ov, nv = old.get(field), new.get(field)
        if ov != nv:
            meta_changes.append({"field": field, "old": ov, "new": nv})

    return {
        "policy_id_old":       old.get("policy_id"),
        "policy_id_new":       new.get("policy_id"),
        "added_drugs":         sorted(new_drugs - old_drugs),
        "removed_drugs":       sorted(old_drugs - new_drugs),
        "unchanged_drugs":     sorted(old_drugs & new_drugs),
        "added_criteria":      sorted(new_crits - old_crits),
        "removed_criteria":    sorted(old_crits - new_crits),
        "metadata_changes":    meta_changes,
        "summary": (
            f"{len(new_drugs - old_drugs)} drug(s) added, "
            f"{len(old_drugs - new_drugs)} drug(s) removed, "
            f"{len(new_crits - old_crits)} criteria added, "
            f"{len(old_crits - new_crits)} criteria removed."
        ),
    }


if __name__ == "__main__":
    mcp.run()
