"""Graph node and relationship models for Neo4j serialisation."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Node models ───────────────────────────────────────────────────────────────
# Each model maps directly to a Neo4j node label.
# The `properties()` method returns the dict passed to CREATE / MERGE.

class NodeBase(BaseModel):
    """Base for all graph node models."""

    def properties(self) -> dict[str, Any]:
        """Return a dict of non-None properties suitable for Cypher parameters."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class PayerNode(NodeBase):
    payer_id: str
    name: str
    type: str
    website: Optional[str] = None

    @property
    def label(self) -> str:
        return "Payer"

    @property
    def merge_key(self) -> dict[str, str]:
        return {"payer_id": self.payer_id}


class PlanNode(NodeBase):
    plan_id: str
    name: str
    payer_id: str
    plan_type: str
    formulary_id: Optional[str] = None

    @property
    def label(self) -> str:
        return "Plan"

    @property
    def merge_key(self) -> dict[str, str]:
        return {"plan_id": self.plan_id}


class PolicyNode(NodeBase):
    policy_id: str
    title: str
    payer_id: str
    effective_date: Optional[str] = None      # ISO date string
    last_reviewed_date: Optional[str] = None  # ISO date string
    version: Optional[str] = None
    source_url: Optional[str] = None
    document_hash: Optional[str] = None

    @property
    def label(self) -> str:
        return "Policy"

    @property
    def merge_key(self) -> dict[str, str]:
        return {"policy_id": self.policy_id}


class DrugNode(NodeBase):
    drug_id: str
    name: str
    generic_name: Optional[str] = None
    rxnorm_cui: Optional[str] = None
    hcpcs_code: Optional[str] = None
    drug_class: Optional[str] = None

    @property
    def label(self) -> str:
        return "Drug"

    @property
    def merge_key(self) -> dict[str, str]:
        return {"drug_id": self.drug_id}


class IndicationNode(NodeBase):
    indication_id: str
    name: str
    description: Optional[str] = None

    @property
    def label(self) -> str:
        return "Indication"

    @property
    def merge_key(self) -> dict[str, str]:
        return {"indication_id": self.indication_id}


class CriterionNode(NodeBase):
    criterion_id: str
    type: str
    description: str
    required_value: Optional[str] = None
    sequence: Optional[int] = None

    @property
    def label(self) -> str:
        return "Criterion"

    @property
    def merge_key(self) -> dict[str, str]:
        return {"criterion_id": self.criterion_id}


class SourceExcerptNode(NodeBase):
    excerpt_id: str
    policy_id: str
    text: str
    page_number: int
    topic: Optional[str] = None

    @property
    def label(self) -> str:
        return "SourceExcerpt"

    @property
    def merge_key(self) -> dict[str, str]:
        return {"excerpt_id": self.excerpt_id}


# ── Relationship models ───────────────────────────────────────────────────────

class RelationshipBase(BaseModel):
    """Base for all graph relationship models."""

    from_id: str = Field(..., description="ID of the source node")
    to_id: str = Field(..., description="ID of the target node")

    def properties(self) -> dict[str, Any]:
        exclude = {"from_id", "to_id"}
        return {k: v for k, v in self.model_dump().items() if v is not None and k not in exclude}


class OffersRelationship(RelationshipBase):
    """(Payer)-[:OFFERS]->(Plan)"""
    type: str = Field("OFFERS", frozen=True)


class HasPolicyRelationship(RelationshipBase):
    """(Plan)-[:HAS_POLICY]->(Policy)"""
    type: str = Field("HAS_POLICY", frozen=True)


class CoversRelationship(RelationshipBase):
    """(Policy)-[:COVERS]->(Drug)"""
    type: str = Field("COVERS", frozen=True)
    coverage_status: Optional[str] = None
    tier: Optional[str] = None


class ExcludesRelationship(RelationshipBase):
    """(Policy)-[:EXCLUDES]->(Drug)"""
    type: str = Field("EXCLUDES", frozen=True)


class AppliesToIndicationRelationship(RelationshipBase):
    """(Policy)-[:APPLIES_TO_INDICATION]->(Indication)"""
    type: str = Field("APPLIES_TO_INDICATION", frozen=True)


class TreatsRelationship(RelationshipBase):
    """(Drug)-[:TREATS]->(Indication)"""
    type: str = Field("TREATS", frozen=True)


class RequiresRelationship(RelationshipBase):
    """(Policy)-[:REQUIRES]->(Criterion)"""
    type: str = Field("REQUIRES", frozen=True)
    for_drug: Optional[str] = None
    for_indication: Optional[str] = None


class CitesRelationship(RelationshipBase):
    """(Policy)-[:CITES]->(SourceExcerpt)"""
    type: str = Field("CITES", frozen=True)


class SupersedesRelationship(RelationshipBase):
    """(Policy)-[:SUPERSEDES]->(Policy)"""
    type: str = Field("SUPERSEDES", frozen=True)
