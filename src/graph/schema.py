"""Neo4j schema — uniqueness constraints and indexes.

Run once via:  python scripts/init_neo4j.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.client import Neo4jClient

logger = logging.getLogger(__name__)

# ── Uniqueness constraints ────────────────────────────────────────────────────
# Each tuple is (label, property).  Neo4j also implicitly creates a b-tree
# index on each constrained property.

UNIQUENESS_CONSTRAINTS: list[tuple[str, str]] = [
    ("Payer",         "payer_id"),
    ("Plan",          "plan_id"),
    ("Policy",        "policy_id"),
    ("Drug",          "drug_id"),
    ("Indication",    "indication_id"),
    ("Criterion",     "criterion_id"),
    ("SourceExcerpt", "excerpt_id"),
]

# ── Extra lookup indexes ───────────────────────────────────────────────────────
# (label, property) pairs that are frequently filtered on but not constrained.

LOOKUP_INDEXES: list[tuple[str, str]] = [
    ("Drug",      "rxnorm_cui"),
    ("Drug",      "hcpcs_code"),
    ("Drug",      "drug_class"),
    ("Drug",      "name"),
    ("Policy",    "payer_id"),
    ("Policy",    "effective_date"),
    ("Payer",     "name"),
    ("Indication","name"),
]

# ── Full-text indexes ─────────────────────────────────────────────────────────
# (index_name, label, property)

FULLTEXT_INDEXES: list[tuple[str, str, str]] = [
    ("drugNameFT",        "Drug",          "name"),
    ("indicationNameFT",  "Indication",    "name"),
    ("policyTitleFT",     "Policy",        "title"),
    ("excerptTextFT",     "SourceExcerpt", "text"),
]


async def init_schema(client: "Neo4jClient") -> None:
    """Create all constraints and indexes idempotently.

    Safe to run multiple times — uses IF NOT EXISTS throughout.
    """
    statements: list[str] = []

    # Uniqueness constraints
    for label, prop in UNIQUENESS_CONSTRAINTS:
        name = f"unique_{label.lower()}_{prop}"
        statements.append(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )

    # Lookup indexes
    for label, prop in LOOKUP_INDEXES:
        name = f"idx_{label.lower()}_{prop}"
        statements.append(
            f"CREATE INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{prop})"
        )

    # Full-text indexes
    for idx_name, label, prop in FULLTEXT_INDEXES:
        statements.append(
            f"CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON EACH [n.{prop}]"
        )

    logger.info("Applying %d schema statements…", len(statements))
    for stmt in statements:
        try:
            await client.run_write(stmt)
            logger.debug("OK: %s", stmt[:80])
        except Exception as exc:
            # Already-existing objects surface as warnings, not hard failures
            logger.warning("Schema statement skipped (%s): %.80s", exc, stmt)

    logger.info("Schema initialisation complete.")


def get_schema_cypher() -> list[str]:
    """Return all schema DDL statements as strings (for inspection / testing)."""
    stmts: list[str] = []
    for label, prop in UNIQUENESS_CONSTRAINTS:
        name = f"unique_{label.lower()}_{prop}"
        stmts.append(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
    for label, prop in LOOKUP_INDEXES:
        name = f"idx_{label.lower()}_{prop}"
        stmts.append(
            f"CREATE INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{prop})"
        )
    for idx_name, label, prop in FULLTEXT_INDEXES:
        stmts.append(
            f"CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON EACH [n.{prop}]"
        )
    return stmts
