"""Pydantic data models for medical benefit drug policy entities."""

from __future__ import annotations

import hashlib
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class PayerType(str, Enum):
    COMMERCIAL = "commercial"
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    EXCHANGE = "exchange"


class PlanType(str, Enum):
    HMO = "hmo"
    PPO = "ppo"
    EPO = "epo"
    POS = "pos"
    HDHP = "hdhp"
    OTHER = "other"


class CriterionType(str, Enum):
    PRIOR_AUTH = "prior_auth"
    STEP_THERAPY = "step_therapy"
    QUANTITY_LIMIT = "quantity_limit"
    AGE_REQUIREMENT = "age_requirement"
    LAB_REQUIREMENT = "lab_requirement"
    DIAGNOSIS_REQUIREMENT = "diagnosis_requirement"
    PRESCRIBER_REQUIREMENT = "prescriber_requirement"
    OTHER = "other"


class CoverageStatus(str, Enum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    COVERED_WITH_RESTRICTIONS = "covered_with_restrictions"
    INVESTIGATIONAL = "investigational"
    EXCLUDED = "excluded"


# ── Core entity models ────────────────────────────────────────────────────────

class Payer(BaseModel):
    """A health insurance payer organisation."""

    payer_id: str = Field(..., description="Unique identifier (e.g. 'aetna')")
    name: str = Field(..., description="Human-readable name (e.g. 'Aetna')")
    type: PayerType = Field(PayerType.COMMERCIAL, description="Payer market segment")
    website: Optional[str] = Field(None, description="Payer's public website URL")

    @field_validator("payer_id")
    @classmethod
    def _slugify(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class Plan(BaseModel):
    """A specific insurance plan offered by a payer."""

    plan_id: str = Field(..., description="Unique identifier for the plan")
    name: str = Field(..., description="Plan marketing name")
    payer_id: str = Field(..., description="Foreign key to Payer.payer_id")
    plan_type: PlanType = Field(PlanType.OTHER)
    formulary_id: Optional[str] = Field(None, description="Associated formulary identifier")


class Policy(BaseModel):
    """A single medical benefit drug policy document."""

    policy_id: str = Field(..., description="Unique identifier (e.g. hash of payer+title+version)")
    title: str = Field(..., description="Policy document title")
    payer_id: str = Field(..., description="Foreign key to Payer.payer_id")
    effective_date: Optional[date] = None
    last_reviewed_date: Optional[date] = None
    version: Optional[str] = None
    source_url: Optional[str] = Field(None, description="URL the PDF was downloaded from")
    document_hash: Optional[str] = Field(None, description="SHA-256 of the raw PDF bytes")
    raw_pdf_path: Optional[str] = Field(None, description="Local path to the source PDF")


class Drug(BaseModel):
    """A pharmaceutical drug or biologic."""

    drug_id: str = Field(..., description="Internal unique ID (derived from rxnorm_cui or name slug)")
    name: str = Field(..., description="Primary name used in the policy")
    generic_name: Optional[str] = None
    brand_names: list[str] = Field(default_factory=list)
    rxnorm_cui: Optional[str] = Field(None, description="RxNorm Concept Unique Identifier")
    hcpcs_code: Optional[str] = Field(None, description="HCPCS J-code or Q-code")
    drug_class: Optional[str] = Field(None, description="Therapeutic class (e.g. 'Anti-VEGF')")
    ndc_codes: list[str] = Field(default_factory=list, description="NDC codes if present")

    @field_validator("drug_id")
    @classmethod
    def _ensure_slug(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class Indication(BaseModel):
    """A clinical indication / diagnosis."""

    indication_id: str = Field(..., description="Slug derived from name")
    name: str = Field(..., description="Indication name (e.g. 'Non-small cell lung cancer')")
    description: Optional[str] = None
    icd10_codes: list[str] = Field(default_factory=list, description="ICD-10-CM codes")

    @field_validator("indication_id")
    @classmethod
    def _ensure_slug(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class Criterion(BaseModel):
    """A single coverage criterion (prior auth, step therapy, etc.)."""

    criterion_id: str = Field(..., description="Unique identifier for this criterion")
    type: CriterionType
    description: str = Field(..., description="Plain-text description of the requirement")
    required_value: Optional[str] = Field(
        None,
        description="Threshold / value if applicable (e.g. 'HbA1c > 7.5%', '30-day supply')",
    )
    applies_to_drug: Optional[str] = Field(None, description="drug_id this criterion targets")
    applies_to_indication: Optional[str] = Field(
        None, description="indication_id this criterion targets"
    )
    sequence: Optional[int] = Field(
        None, description="Order in a step-therapy sequence (1 = first-line)"
    )


class SourceExcerpt(BaseModel):
    """A verbatim excerpt from a policy document, with location metadata."""

    excerpt_id: str = Field(..., description="Unique identifier (hash of policy_id+page+text)")
    policy_id: str = Field(..., description="Foreign key to Policy.policy_id")
    text: str = Field(..., description="Verbatim text extracted from the page")
    page_number: int = Field(..., ge=1)
    bbox: Optional[list[float]] = Field(
        None, description="Bounding box [x0, y0, x1, y1] in PDF points"
    )
    topic: Optional[str] = Field(
        None, description="Semantic topic label (e.g. 'prior_auth', 'exclusion')"
    )

    @computed_field  # type: ignore[misc]
    @property
    def _auto_id(self) -> str:
        """Auto-generate excerpt_id from content if not supplied."""
        raw = f"{self.policy_id}:{self.page_number}:{self.text[:200]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Coverage fact (policy × drug relationship) ────────────────────────────────

class CoverageFact(BaseModel):
    """Records the coverage status of a drug under a specific policy."""

    policy_id: str
    drug_id: str
    coverage_status: CoverageStatus = CoverageStatus.COVERED_WITH_RESTRICTIONS
    tier: Optional[str] = Field(None, description="Formulary tier (e.g. 'Tier 3')")
    notes: Optional[str] = None


# ── Composite extraction output ───────────────────────────────────────────────

class ExtractedPolicy(BaseModel):
    """Top-level output produced by the VLM extractor for a single PDF.

    This is the contract between the ingestion pipeline and the graph builder.
    Every sub-object uses the models above so the graph builder can work
    purely from this structure.
    """

    policy: Policy
    payer: Payer
    plans: list[Plan] = Field(default_factory=list)
    drugs: list[Drug] = Field(default_factory=list)
    indications: list[Indication] = Field(default_factory=list)
    criteria: list[Criterion] = Field(default_factory=list)
    coverage_facts: list[CoverageFact] = Field(default_factory=list)
    excerpts: list[SourceExcerpt] = Field(default_factory=list)

    # Extraction provenance
    extractor_model: Optional[str] = Field(
        None, description="Claude model ID used for extraction"
    )
    extraction_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Self-reported extraction confidence score"
    )
    extraction_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues noted during extraction (e.g. blurry page)",
    )

    def drug_by_name(self, name: str) -> Optional[Drug]:
        """Return the first drug whose name or generic_name matches (case-insensitive)."""
        name_lower = name.lower()
        for drug in self.drugs:
            if drug.name.lower() == name_lower:
                return drug
            if drug.generic_name and drug.generic_name.lower() == name_lower:
                return drug
        return None

    def criteria_for_drug(self, drug_id: str) -> list[Criterion]:
        """Return all criteria that apply to the given drug_id."""
        return [c for c in self.criteria if c.applies_to_drug == drug_id]
