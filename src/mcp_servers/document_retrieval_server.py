"""Document Retrieval MCP server.

Exposes semantic search over policy excerpts and policy metadata lookups
so Claude can cite specific pages when answering questions.

Run standalone:
    python scripts/run_mcp_server.py document_retrieval
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── Lazy clients ──────────────────────────────────────────────────────────────

_neo4j_client: Any = None
_excerpt_store: Any = None


async def _get_neo4j() -> Any:
    global _neo4j_client
    if _neo4j_client is None:
        from src.graph.client import Neo4jClient
        _neo4j_client = Neo4jClient.from_settings()
        await _neo4j_client.connect()
    return _neo4j_client


def _get_store() -> Any:
    global _excerpt_store
    if _excerpt_store is None:
        from src.vector_store.excerpt_store import ExcerptStore
        _excerpt_store = ExcerptStore.from_settings()
    return _excerpt_store


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncGenerator[None, None]:
    yield
    if _neo4j_client is not None:
        await _neo4j_client.close()


# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "document-retrieval",
    instructions=(
        "Retrieve verbatim excerpts from medical benefit drug policy PDFs "
        "with exact page number citations, and look up policy metadata."
    ),
    lifespan=_lifespan,
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_policy_excerpt(policy_id: str, topic: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant verbatim excerpts from a policy document.

    Uses semantic search to find the most relevant passages for a topic,
    returning the text with exact page numbers for citation.

    Args:
        policy_id: The policy_id to search within.
        topic: Free-text topic or question (e.g. "prior authorization criteria", "step therapy").
        n_results: Maximum number of excerpts to return (default 5).

    Returns:
        List of {excerpt_id, text, page_number, topic, relevance, citation}.
    """
    store = _get_store()
    results = store.search(topic, policy_id=policy_id, n_results=n_results)
    if not results:
        return [{"message": f"No excerpts found for policy '{policy_id}' on topic '{topic}'."}]
    return [
        {
            "excerpt_id":  r.excerpt_id,
            "text":        r.text,
            "page_number": r.page_number,
            "topic":       r.topic,
            "relevance":   round(r.relevance, 4),
            "citation":    r.citation,
        }
        for r in results
    ]


@mcp.tool()
async def get_policy_metadata(policy_id: str) -> dict[str, Any]:
    """Return metadata for a single policy document.

    Args:
        policy_id: The unique policy identifier.

    Returns:
        Dict with title, payer, effective_date, version, source_url, covered drugs,
        indications, and criteria summary.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    result = await queries.get_policy_details(neo4j, policy_id)
    if not result:
        return {"error": f"Policy '{policy_id}' not found."}
    return result


@mcp.tool()
async def list_policies_for_drug(drug_name: str) -> list[dict[str, Any]]:
    """Find all policies that mention (cover or exclude) a drug.

    Args:
        drug_name: Generic or brand name of the drug.

    Returns:
        List of {policy_id, title, effective_date, payer, relationship}
        where relationship is 'COVERS' or 'EXCLUDES'.
    """
    from src.graph import queries
    neo4j = await _get_neo4j()
    results = await queries.list_policies_for_drug(neo4j, drug_name)
    if not results:
        return [{"message": f"No policies found mentioning '{drug_name}'."}]
    return results


if __name__ == "__main__":
    mcp.run()
