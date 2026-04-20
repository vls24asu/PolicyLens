"""Parameterised Cypher query functions.

All public functions accept a Neo4jClient and typed parameters; they never
interpolate user-supplied strings into Cypher — values go through the
parameters dict only.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.client import Neo4jClient

logger = logging.getLogger(__name__)


# ── Write helpers — node upserts ──────────────────────────────────────────────

async def upsert_payer(client: "Neo4jClient", props: dict[str, Any]) -> None:
    """MERGE a Payer node."""
    await client.run_write(
        """
        MERGE (p:Payer {payer_id: $payer_id})
        SET p += $props
        """,
        {"payer_id": props["payer_id"], "props": props},
    )


async def upsert_plan(client: "Neo4jClient", props: dict[str, Any]) -> None:
    """MERGE a Plan node."""
    await client.run_write(
        """
        MERGE (pl:Plan {plan_id: $plan_id})
        SET pl += $props
        """,
        {"plan_id": props["plan_id"], "props": props},
    )


async def upsert_policy(client: "Neo4jClient", props: dict[str, Any]) -> None:
    """MERGE a Policy node."""
    await client.run_write(
        """
        MERGE (po:Policy {policy_id: $policy_id})
        SET po += $props
        """,
        {"policy_id": props["policy_id"], "props": props},
    )


async def upsert_drug(client: "Neo4jClient", props: dict[str, Any]) -> None:
    """MERGE a Drug node."""
    await client.run_write(
        """
        MERGE (d:Drug {drug_id: $drug_id})
        SET d += $props
        """,
        {"drug_id": props["drug_id"], "props": props},
    )


async def upsert_indication(client: "Neo4jClient", props: dict[str, Any]) -> None:
    """MERGE an Indication node."""
    await client.run_write(
        """
        MERGE (i:Indication {indication_id: $indication_id})
        SET i += $props
        """,
        {"indication_id": props["indication_id"], "props": props},
    )


async def upsert_criterion(client: "Neo4jClient", props: dict[str, Any]) -> None:
    """MERGE a Criterion node."""
    await client.run_write(
        """
        MERGE (c:Criterion {criterion_id: $criterion_id})
        SET c += $props
        """,
        {"criterion_id": props["criterion_id"], "props": props},
    )


async def upsert_excerpt(client: "Neo4jClient", props: dict[str, Any]) -> None:
    """MERGE a SourceExcerpt node."""
    await client.run_write(
        """
        MERGE (e:SourceExcerpt {excerpt_id: $excerpt_id})
        SET e += $props
        """,
        {"excerpt_id": props["excerpt_id"], "props": props},
    )


# ── Write helpers — relationships ─────────────────────────────────────────────

async def link_payer_plan(client: "Neo4jClient", payer_id: str, plan_id: str) -> None:
    """(Payer)-[:OFFERS]->(Plan)"""
    await client.run_write(
        """
        MATCH (a:Payer  {payer_id: $payer_id})
        MATCH (b:Plan   {plan_id:  $plan_id})
        MERGE (a)-[:OFFERS]->(b)
        """,
        {"payer_id": payer_id, "plan_id": plan_id},
    )


async def link_plan_policy(client: "Neo4jClient", plan_id: str, policy_id: str) -> None:
    """(Plan)-[:HAS_POLICY]->(Policy)"""
    await client.run_write(
        """
        MATCH (a:Plan   {plan_id:   $plan_id})
        MATCH (b:Policy {policy_id: $policy_id})
        MERGE (a)-[:HAS_POLICY]->(b)
        """,
        {"plan_id": plan_id, "policy_id": policy_id},
    )


async def link_policy_covers_drug(
    client: "Neo4jClient",
    policy_id: str,
    drug_id: str,
    coverage_status: str,
    tier: str | None = None,
) -> None:
    """(Policy)-[:COVERS {coverage_status, tier}]->(Drug)"""
    await client.run_write(
        """
        MATCH (a:Policy {policy_id: $policy_id})
        MATCH (b:Drug   {drug_id:   $drug_id})
        MERGE (a)-[r:COVERS]->(b)
        SET r.coverage_status = $coverage_status,
            r.tier             = $tier
        """,
        {
            "policy_id":       policy_id,
            "drug_id":         drug_id,
            "coverage_status": coverage_status,
            "tier":            tier,
        },
    )


async def link_policy_excludes_drug(
    client: "Neo4jClient", policy_id: str, drug_id: str
) -> None:
    """(Policy)-[:EXCLUDES]->(Drug)"""
    await client.run_write(
        """
        MATCH (a:Policy {policy_id: $policy_id})
        MATCH (b:Drug   {drug_id:   $drug_id})
        MERGE (a)-[:EXCLUDES]->(b)
        """,
        {"policy_id": policy_id, "drug_id": drug_id},
    )


async def link_policy_indication(
    client: "Neo4jClient", policy_id: str, indication_id: str
) -> None:
    """(Policy)-[:APPLIES_TO_INDICATION]->(Indication)"""
    await client.run_write(
        """
        MATCH (a:Policy    {policy_id:     $policy_id})
        MATCH (b:Indication{indication_id: $indication_id})
        MERGE (a)-[:APPLIES_TO_INDICATION]->(b)
        """,
        {"policy_id": policy_id, "indication_id": indication_id},
    )


async def link_drug_treats(
    client: "Neo4jClient", drug_id: str, indication_id: str
) -> None:
    """(Drug)-[:TREATS]->(Indication)"""
    await client.run_write(
        """
        MATCH (a:Drug      {drug_id:       $drug_id})
        MATCH (b:Indication{indication_id: $indication_id})
        MERGE (a)-[:TREATS]->(b)
        """,
        {"drug_id": drug_id, "indication_id": indication_id},
    )


async def link_policy_requires(
    client: "Neo4jClient",
    policy_id: str,
    criterion_id: str,
    for_drug: str | None = None,
    for_indication: str | None = None,
) -> None:
    """(Policy)-[:REQUIRES {for_drug, for_indication}]->(Criterion)"""
    await client.run_write(
        """
        MATCH (a:Policy   {policy_id:    $policy_id})
        MATCH (b:Criterion{criterion_id: $criterion_id})
        MERGE (a)-[r:REQUIRES]->(b)
        SET r.for_drug        = $for_drug,
            r.for_indication  = $for_indication
        """,
        {
            "policy_id":      policy_id,
            "criterion_id":   criterion_id,
            "for_drug":       for_drug,
            "for_indication": for_indication,
        },
    )


async def link_policy_cites(
    client: "Neo4jClient", policy_id: str, excerpt_id: str
) -> None:
    """(Policy)-[:CITES]->(SourceExcerpt)"""
    await client.run_write(
        """
        MATCH (a:Policy       {policy_id:  $policy_id})
        MATCH (b:SourceExcerpt{excerpt_id: $excerpt_id})
        MERGE (a)-[:CITES]->(b)
        """,
        {"policy_id": policy_id, "excerpt_id": excerpt_id},
    )


async def link_policy_supersedes(
    client: "Neo4jClient", new_policy_id: str, old_policy_id: str
) -> None:
    """(Policy)-[:SUPERSEDES]->(Policy)"""
    await client.run_write(
        """
        MATCH (a:Policy {policy_id: $new_id})
        MATCH (b:Policy {policy_id: $old_id})
        MERGE (a)-[:SUPERSEDES]->(b)
        """,
        {"new_id": new_policy_id, "old_id": old_policy_id},
    )


# ── Read queries ──────────────────────────────────────────────────────────────

async def query_coverage(
    client: "Neo4jClient",
    drug_name: str,
    indication_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return plans that cover a drug, optionally filtered by indication.

    Returns a list of dicts with keys: payer, plan, policy, coverage_status, tier.
    """
    if indication_name:
        cypher = """
        MATCH (payer:Payer)-[:OFFERS]->(plan:Plan)-[:HAS_POLICY]->(pol:Policy)
              -[cov:COVERS]->(drug:Drug)
        MATCH (pol)-[:APPLIES_TO_INDICATION]->(ind:Indication)
        WHERE toLower(drug.name) CONTAINS toLower($drug_name)
          AND toLower(ind.name)  CONTAINS toLower($indication_name)
        RETURN payer.name       AS payer,
               plan.name        AS plan,
               pol.policy_id    AS policy_id,
               pol.title        AS policy_title,
               cov.coverage_status AS coverage_status,
               cov.tier         AS tier
        ORDER BY payer.name, plan.name
        """
        params: dict[str, Any] = {"drug_name": drug_name, "indication_name": indication_name}
    else:
        cypher = """
        MATCH (payer:Payer)-[:OFFERS]->(plan:Plan)-[:HAS_POLICY]->(pol:Policy)
              -[cov:COVERS]->(drug:Drug)
        WHERE toLower(drug.name) CONTAINS toLower($drug_name)
        RETURN payer.name       AS payer,
               plan.name        AS plan,
               pol.policy_id    AS policy_id,
               pol.title        AS policy_title,
               cov.coverage_status AS coverage_status,
               cov.tier         AS tier
        ORDER BY payer.name, plan.name
        """
        params = {"drug_name": drug_name}

    return await client.run(cypher, params)


async def get_prior_auth_criteria(
    client: "Neo4jClient", drug_name: str, plan_name: str
) -> list[dict[str, Any]]:
    """Return prior-auth criteria for a drug under a specific plan."""
    cypher = """
    MATCH (plan:Plan {name: $plan_name})-[:HAS_POLICY]->(pol:Policy)
    MATCH (pol)-[req:REQUIRES]->(crit:Criterion)
    MATCH (drug:Drug)
    WHERE toLower(drug.name) CONTAINS toLower($drug_name)
      AND (req.for_drug = drug.drug_id OR req.for_drug IS NULL)
      AND crit.type IN ['prior_auth', 'step_therapy', 'quantity_limit',
                        'age_requirement', 'lab_requirement']
    RETURN crit.criterion_id  AS criterion_id,
           crit.type           AS type,
           crit.description    AS description,
           crit.required_value AS required_value,
           crit.sequence       AS sequence,
           pol.policy_id       AS policy_id
    ORDER BY crit.sequence, crit.type
    """
    return await client.run(cypher, {"drug_name": drug_name, "plan_name": plan_name})


async def compare_policies(
    client: "Neo4jClient", drug_name: str, plan_a: str, plan_b: str
) -> dict[str, Any]:
    """Return a structured comparison of coverage between two plans for a drug."""

    async def _fetch(plan: str) -> list[dict[str, Any]]:
        cypher = """
        MATCH (plan:Plan)-[:HAS_POLICY]->(pol:Policy)-[cov:COVERS]->(drug:Drug)
        WHERE plan.name = $plan_name
          AND toLower(drug.name) CONTAINS toLower($drug_name)
        OPTIONAL MATCH (pol)-[req:REQUIRES]->(crit:Criterion)
        WHERE req.for_drug = drug.drug_id OR req.for_drug IS NULL
        RETURN pol.policy_id       AS policy_id,
               pol.title           AS policy_title,
               pol.effective_date  AS effective_date,
               cov.coverage_status AS coverage_status,
               cov.tier            AS tier,
               collect({
                 type:           crit.type,
                 description:    crit.description,
                 required_value: crit.required_value
               })                  AS criteria
        """
        return await client.run(cypher, {"plan_name": plan, "drug_name": drug_name})

    rows_a, rows_b = await _fetch(plan_a), await _fetch(plan_b)
    return {
        "drug": drug_name,
        plan_a: rows_a[0] if rows_a else None,
        plan_b: rows_b[0] if rows_b else None,
    }


async def search_policies_by_drug_class(
    client: "Neo4jClient", drug_class: str
) -> list[dict[str, Any]]:
    """Return policies that cover any drug in the given therapeutic class."""
    cypher = """
    MATCH (pol:Policy)-[:COVERS]->(drug:Drug)
    WHERE toLower(drug.drug_class) CONTAINS toLower($drug_class)
    MATCH (payer:Payer)-[:OFFERS]->(:Plan)-[:HAS_POLICY]->(pol)
    RETURN DISTINCT pol.policy_id   AS policy_id,
           pol.title                AS title,
           pol.effective_date       AS effective_date,
           payer.name               AS payer,
           collect(DISTINCT drug.name) AS drugs
    ORDER BY payer.name, pol.title
    """
    return await client.run(cypher, {"drug_class": drug_class})


async def list_payers(client: "Neo4jClient") -> list[dict[str, Any]]:
    """Return all payers stored in the database."""
    cypher = """
    MATCH (p:Payer)
    OPTIONAL MATCH (p)-[:OFFERS]->(pl:Plan)
    OPTIONAL MATCH (pl)-[:HAS_POLICY]->(po:Policy)
    RETURN p.payer_id           AS payer_id,
           p.name               AS name,
           p.type               AS type,
           count(DISTINCT pl)   AS plan_count,
           count(DISTINCT po)   AS policy_count
    ORDER BY p.name
    """
    return await client.run(cypher)


async def list_policies(
    client: "Neo4jClient",
    payer_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List policies, optionally filtered by payer."""
    if payer_id:
        cypher = """
        MATCH (payer:Payer {payer_id: $payer_id})-[:OFFERS]->(:Plan)-[:HAS_POLICY]->(pol:Policy)
        RETURN DISTINCT pol.policy_id      AS policy_id,
               pol.title                   AS title,
               pol.effective_date          AS effective_date,
               pol.version                 AS version,
               payer.name                  AS payer
        ORDER BY pol.effective_date DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {"payer_id": payer_id, "limit": limit}
    else:
        cypher = """
        MATCH (payer:Payer)-[:OFFERS]->(:Plan)-[:HAS_POLICY]->(pol:Policy)
        RETURN DISTINCT pol.policy_id      AS policy_id,
               pol.title                   AS title,
               pol.effective_date          AS effective_date,
               pol.version                 AS version,
               payer.name                  AS payer
        ORDER BY pol.effective_date DESC
        LIMIT $limit
        """
        params = {"limit": limit}
    return await client.run(cypher, params)


async def get_policy_details(
    client: "Neo4jClient", policy_id: str
) -> dict[str, Any] | None:
    """Return full details for a single policy."""
    cypher = """
    MATCH (pol:Policy {policy_id: $policy_id})
    OPTIONAL MATCH (pol)-[:COVERS]->(drug:Drug)
    OPTIONAL MATCH (pol)-[:APPLIES_TO_INDICATION]->(ind:Indication)
    OPTIONAL MATCH (pol)-[:REQUIRES]->(crit:Criterion)
    RETURN pol.policy_id        AS policy_id,
           pol.title             AS title,
           pol.effective_date    AS effective_date,
           pol.last_reviewed_date AS last_reviewed_date,
           pol.version           AS version,
           pol.source_url        AS source_url,
           collect(DISTINCT {
             drug_id:   drug.drug_id,
             name:      drug.name,
             drug_class: drug.drug_class
           })                    AS drugs,
           collect(DISTINCT {
             indication_id: ind.indication_id,
             name:          ind.name
           })                    AS indications,
           collect(DISTINCT {
             criterion_id:   crit.criterion_id,
             type:           crit.type,
             description:    crit.description,
             required_value: crit.required_value
           })                    AS criteria
    """
    rows = await client.run(cypher, {"policy_id": policy_id})
    return rows[0] if rows else None


async def get_recent_policy_changes(
    client: "Neo4jClient",
    payer_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Return policies whose effective_date falls in the given range."""
    filters = []
    params: dict[str, Any] = {}
    if payer_id:
        filters.append("payer.payer_id = $payer_id")
        params["payer_id"] = payer_id
    if start_date:
        filters.append("pol.effective_date >= $start_date")
        params["start_date"] = start_date
    if end_date:
        filters.append("pol.effective_date <= $end_date")
        params["end_date"] = end_date

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    cypher = f"""
    MATCH (payer:Payer)-[:OFFERS]->(:Plan)-[:HAS_POLICY]->(pol:Policy)
    {where_clause}
    RETURN DISTINCT pol.policy_id     AS policy_id,
           pol.title                  AS title,
           pol.effective_date         AS effective_date,
           pol.version                AS version,
           payer.name                 AS payer
    ORDER BY pol.effective_date DESC
    """
    return await client.run(cypher, params)


async def list_policies_for_drug(
    client: "Neo4jClient", drug_name: str
) -> list[dict[str, Any]]:
    """Return all policies that mention (cover or exclude) a drug."""
    cypher = """
    MATCH (drug:Drug)
    WHERE toLower(drug.name) CONTAINS toLower($drug_name)
    MATCH (pol:Policy)-[r:COVERS|EXCLUDES]->(drug)
    MATCH (payer:Payer)-[:OFFERS]->(:Plan)-[:HAS_POLICY]->(pol)
    RETURN DISTINCT pol.policy_id   AS policy_id,
           pol.title                AS title,
           pol.effective_date       AS effective_date,
           payer.name               AS payer,
           type(r)                  AS relationship
    ORDER BY payer.name, pol.title
    """
    return await client.run(cypher, {"drug_name": drug_name})
