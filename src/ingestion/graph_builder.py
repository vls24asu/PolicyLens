"""Populate Neo4j from an ExtractedPolicy.

GraphBuilder is the bridge between the VLM extraction pipeline and the graph
database.  It converts every entity in an ExtractedPolicy into graph nodes and
relationships using the parameterised query helpers in src.graph.queries.

Design notes
------------
- All node upserts run first so relationships never reference missing nodes.
- Independent upserts within the same phase are gathered concurrently.
- Relationship creation runs after all nodes are committed.
- Every public method is idempotent — re-ingesting the same policy is safe.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

from src.graph import queries
from src.graph.client import Neo4jClient
from src.models.policy import (
    CoverageStatus,
    ExtractedPolicy,
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Write an ExtractedPolicy to Neo4j.

    Parameters
    ----------
    client:
        A connected ``Neo4jClient`` instance.
    """

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    # ── Public API ────────────────────────────────────────────────────────────

    async def build(self, extracted: ExtractedPolicy) -> None:
        """Ingest a fully-extracted policy into the graph.

        Steps
        -----
        1. Upsert all nodes (payer, plans, policy, drugs, indications, criteria, excerpts)
        2. Create all relationships between those nodes

        Parameters
        ----------
        extracted:
            Validated ``ExtractedPolicy`` produced by ``VLMExtractor``.
        """
        policy_id = extracted.policy.policy_id
        logger.info("GraphBuilder: ingesting policy %s", policy_id)

        await self._upsert_nodes(extracted)
        await self._create_relationships(extracted)

        logger.info("GraphBuilder: done — policy %s is in the graph.", policy_id)

    # ── Phase 1: node upserts ─────────────────────────────────────────────────

    async def _upsert_nodes(self, extracted: ExtractedPolicy) -> None:
        """Upsert every entity as a graph node."""

        # Payer must exist before plans; everything else is independent.
        await queries.upsert_payer(self._client, _payer_props(extracted))

        await asyncio.gather(
            *[queries.upsert_plan(self._client, _plan_props(plan, extracted.payer.payer_id))
              for plan in extracted.plans],
            queries.upsert_policy(self._client, _policy_props(extracted)),
            *[queries.upsert_drug(self._client, _drug_props(drug))
              for drug in extracted.drugs],
            *[queries.upsert_indication(self._client, _indication_props(ind))
              for ind in extracted.indications],
            *[queries.upsert_criterion(self._client, _criterion_props(crit))
              for crit in extracted.criteria],
            *[queries.upsert_excerpt(self._client, _excerpt_props(exc))
              for exc in extracted.excerpts],
        )

        logger.debug(
            "  Upserted: 1 payer, %d plans, 1 policy, %d drugs, "
            "%d indications, %d criteria, %d excerpts",
            len(extracted.plans), len(extracted.drugs),
            len(extracted.indications), len(extracted.criteria),
            len(extracted.excerpts),
        )

    # ── Phase 2: relationship creation ────────────────────────────────────────

    async def _create_relationships(self, extracted: ExtractedPolicy) -> None:
        """Wire every relationship defined by the graph schema."""
        policy_id = extracted.policy.policy_id
        payer_id  = extracted.payer.payer_id

        tasks: list[Any] = []

        # (Payer)-[:OFFERS]->(Plan)
        for plan in extracted.plans:
            tasks.append(queries.link_payer_plan(self._client, payer_id, plan.plan_id))

        # (Plan)-[:HAS_POLICY]->(Policy)
        for plan in extracted.plans:
            tasks.append(queries.link_plan_policy(self._client, plan.plan_id, policy_id))

        # (Policy)-[:COVERS | :EXCLUDES]->(Drug)
        drug_ids = {d.drug_id for d in extracted.drugs}
        covered_drug_ids: set[str] = set()

        for fact in extracted.coverage_facts:
            if fact.drug_id not in drug_ids:
                logger.warning(
                    "  Coverage fact references unknown drug_id '%s' — skipping.", fact.drug_id
                )
                continue
            if fact.coverage_status == CoverageStatus.EXCLUDED:
                tasks.append(
                    queries.link_policy_excludes_drug(self._client, policy_id, fact.drug_id)
                )
            else:
                covered_drug_ids.add(fact.drug_id)
                tasks.append(
                    queries.link_policy_covers_drug(
                        self._client,
                        policy_id,
                        fact.drug_id,
                        fact.coverage_status.value,
                        fact.tier,
                    )
                )

        # If there are drugs but no coverage_facts, assume all are covered with restrictions
        if not extracted.coverage_facts and extracted.drugs:
            logger.debug("  No coverage_facts — defaulting all drugs to covered_with_restrictions")
            for drug in extracted.drugs:
                tasks.append(
                    queries.link_policy_covers_drug(
                        self._client, policy_id, drug.drug_id,
                        CoverageStatus.COVERED_WITH_RESTRICTIONS.value,
                        None,
                    )
                )

        # (Policy)-[:APPLIES_TO_INDICATION]->(Indication)
        indication_ids = {i.indication_id for i in extracted.indications}
        for indication in extracted.indications:
            tasks.append(
                queries.link_policy_indication(self._client, policy_id, indication.indication_id)
            )

        # (Drug)-[:TREATS]->(Indication)
        # Infer from criteria that link both a drug and an indication.
        seen_treats: set[tuple[str, str]] = set()
        for crit in extracted.criteria:
            if crit.applies_to_drug and crit.applies_to_indication:
                if (crit.applies_to_drug not in drug_ids or
                        crit.applies_to_indication not in indication_ids):
                    continue
                pair = (crit.applies_to_drug, crit.applies_to_indication)
                if pair not in seen_treats:
                    seen_treats.add(pair)
                    tasks.append(
                        queries.link_drug_treats(
                            self._client, crit.applies_to_drug, crit.applies_to_indication
                        )
                    )

        # (Policy)-[:REQUIRES {for_drug, for_indication}]->(Criterion)
        for crit in extracted.criteria:
            tasks.append(
                queries.link_policy_requires(
                    self._client,
                    policy_id,
                    crit.criterion_id,
                    for_drug=crit.applies_to_drug,
                    for_indication=crit.applies_to_indication,
                )
            )

        # (Policy)-[:CITES]->(SourceExcerpt)
        for exc in extracted.excerpts:
            tasks.append(queries.link_policy_cites(self._client, policy_id, exc.excerpt_id))

        await asyncio.gather(*tasks)

        logger.debug(
            "  Relationships: %d plans linked, %d drugs covered, "
            "%d indications, %d drug-treats, %d criteria, %d excerpts",
            len(extracted.plans),
            len(extracted.coverage_facts) or len(extracted.drugs),
            len(extracted.indications),
            len(seen_treats),
            len(extracted.criteria),
            len(extracted.excerpts),
        )

    # ── Supersedes helper (called explicitly by the ingest pipeline) ──────────

    async def mark_supersedes(self, new_policy_id: str, old_policy_id: str) -> None:
        """Record that new_policy_id supersedes old_policy_id."""
        await queries.link_policy_supersedes(self._client, new_policy_id, old_policy_id)
        logger.info("  %s supersedes %s", new_policy_id, old_policy_id)


# ── Property converters (Pydantic model → flat dict for Cypher) ───────────────

def _payer_props(extracted: ExtractedPolicy) -> dict[str, Any]:
    p = extracted.payer
    props: dict[str, Any] = {"payer_id": p.payer_id, "name": p.name, "type": p.type.value}
    if p.website:
        props["website"] = p.website
    return props


def _plan_props(plan: Any, payer_id: str) -> dict[str, Any]:
    props: dict[str, Any] = {
        "plan_id":   plan.plan_id,
        "name":      plan.name,
        "payer_id":  payer_id,
        "plan_type": plan.plan_type.value,
    }
    if plan.formulary_id:
        props["formulary_id"] = plan.formulary_id
    return props


def _policy_props(extracted: ExtractedPolicy) -> dict[str, Any]:
    pol = extracted.policy
    props: dict[str, Any] = {
        "policy_id": pol.policy_id,
        "title":     pol.title,
        "payer_id":  pol.payer_id,
    }
    if pol.effective_date:
        props["effective_date"] = _date_str(pol.effective_date)
    if pol.last_reviewed_date:
        props["last_reviewed_date"] = _date_str(pol.last_reviewed_date)
    if pol.version:
        props["version"] = pol.version
    if pol.source_url:
        props["source_url"] = pol.source_url
    if pol.document_hash:
        props["document_hash"] = pol.document_hash
    return props


def _drug_props(drug: Any) -> dict[str, Any]:
    props: dict[str, Any] = {"drug_id": drug.drug_id, "name": drug.name}
    if drug.generic_name:
        props["generic_name"] = drug.generic_name
    if drug.brand_names:
        props["brand_names"] = drug.brand_names
    if drug.rxnorm_cui:
        props["rxnorm_cui"] = drug.rxnorm_cui
    if drug.hcpcs_code:
        props["hcpcs_code"] = drug.hcpcs_code
    if drug.drug_class:
        props["drug_class"] = drug.drug_class
    if drug.ndc_codes:
        props["ndc_codes"] = drug.ndc_codes
    return props


def _indication_props(ind: Any) -> dict[str, Any]:
    props: dict[str, Any] = {"indication_id": ind.indication_id, "name": ind.name}
    if ind.description:
        props["description"] = ind.description
    if ind.icd10_codes:
        props["icd10_codes"] = ind.icd10_codes
    return props


def _criterion_props(crit: Any) -> dict[str, Any]:
    props: dict[str, Any] = {
        "criterion_id": crit.criterion_id,
        "type":         crit.type.value,
        "description":  crit.description,
    }
    if crit.required_value:
        props["required_value"] = crit.required_value
    if crit.sequence is not None:
        props["sequence"] = crit.sequence
    return props


def _excerpt_props(exc: Any) -> dict[str, Any]:
    props: dict[str, Any] = {
        "excerpt_id":   exc.excerpt_id,
        "policy_id":    exc.policy_id,
        "text":         exc.text,
        "page_number":  exc.page_number,
    }
    if exc.topic:
        props["topic"] = exc.topic
    if exc.bbox:
        props["bbox"] = exc.bbox
    return props


def _date_str(d: date | None) -> str | None:
    return d.isoformat() if d else None
