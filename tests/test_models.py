"""Tests for Pydantic data models (Stage 2)."""

from datetime import date

import pytest

from src.models.policy import (
    CoverageFact,
    CoverageStatus,
    Criterion,
    CriterionType,
    Drug,
    ExtractedPolicy,
    Indication,
    Payer,
    PayerType,
    Plan,
    PlanType,
    Policy,
    SourceExcerpt,
)
from src.models.graph import (
    CoversRelationship,
    DrugNode,
    PolicyNode,
    RequiresRelationship,
)


# ── Payer ──────────────────────────────────────────────────────────────────────

def test_payer_slug_normalisation() -> None:
    payer = Payer(payer_id="Aetna Inc", name="Aetna", type=PayerType.COMMERCIAL)
    assert payer.payer_id == "aetna_inc"


def test_payer_defaults() -> None:
    payer = Payer(payer_id="cigna", name="Cigna")
    assert payer.type == PayerType.COMMERCIAL
    assert payer.website is None


# ── Plan ──────────────────────────────────────────────────────────────────────

def test_plan_model() -> None:
    plan = Plan(plan_id="cigna_ppo_2024", name="Cigna PPO 2024", payer_id="cigna", plan_type=PlanType.PPO)
    assert plan.plan_type == PlanType.PPO


# ── Policy ────────────────────────────────────────────────────────────────────

def test_policy_optional_fields() -> None:
    policy = Policy(policy_id="pol_001", title="Biologic PA Policy", payer_id="aetna")
    assert policy.effective_date is None
    assert policy.document_hash is None


def test_policy_with_dates() -> None:
    policy = Policy(
        policy_id="pol_002",
        title="GLP-1 Coverage Policy",
        payer_id="humana",
        effective_date=date(2024, 1, 1),
        last_reviewed_date=date(2024, 6, 1),
        version="2.3",
    )
    assert policy.version == "2.3"
    assert policy.effective_date == date(2024, 1, 1)


# ── Drug ──────────────────────────────────────────────────────────────────────

def test_drug_slug_normalisation() -> None:
    drug = Drug(drug_id="Pembrolizumab", name="Pembrolizumab", rxnorm_cui="1547545")
    assert drug.drug_id == "pembrolizumab"


def test_drug_brand_names() -> None:
    drug = Drug(
        drug_id="adalimumab",
        name="adalimumab",
        brand_names=["Humira", "Hadlima", "Hyrimoz"],
        drug_class="TNF inhibitor",
    )
    assert len(drug.brand_names) == 3
    assert "Humira" in drug.brand_names


# ── Indication ────────────────────────────────────────────────────────────────

def test_indication_icd10_codes() -> None:
    ind = Indication(
        indication_id="nsclc",
        name="Non-small cell lung cancer",
        icd10_codes=["C34.10", "C34.11", "C34.12"],
    )
    assert "C34.10" in ind.icd10_codes


# ── Criterion ────────────────────────────────────────────────────────────────

def test_criterion_step_therapy() -> None:
    crit = Criterion(
        criterion_id="crit_001",
        type=CriterionType.STEP_THERAPY,
        description="Must fail methotrexate before biologic approval",
        required_value="≥3 months methotrexate trial",
        sequence=1,
    )
    assert crit.type == CriterionType.STEP_THERAPY
    assert crit.sequence == 1


# ── SourceExcerpt ─────────────────────────────────────────────────────────────

def test_source_excerpt() -> None:
    exc = SourceExcerpt(
        excerpt_id="exc_abc123",
        policy_id="pol_001",
        text="Prior authorization is required for all biologic agents.",
        page_number=3,
        topic="prior_auth",
    )
    assert exc.page_number == 3


def test_source_excerpt_page_ge_1() -> None:
    with pytest.raises(Exception):
        SourceExcerpt(
            excerpt_id="exc_bad",
            policy_id="pol_001",
            text="some text",
            page_number=0,
        )


# ── CoverageFact ──────────────────────────────────────────────────────────────

def test_coverage_fact_default_status() -> None:
    fact = CoverageFact(policy_id="pol_001", drug_id="adalimumab")
    assert fact.coverage_status == CoverageStatus.COVERED_WITH_RESTRICTIONS


# ── ExtractedPolicy ───────────────────────────────────────────────────────────

def _make_extracted() -> ExtractedPolicy:
    payer = Payer(payer_id="aetna", name="Aetna")
    policy = Policy(policy_id="pol_001", title="Biologic PA", payer_id="aetna")
    drug = Drug(drug_id="adalimumab", name="adalimumab", generic_name="adalimumab")
    indication = Indication(indication_id="ra", name="Rheumatoid Arthritis", icd10_codes=["M05.79"])
    criterion = Criterion(
        criterion_id="crit_001",
        type=CriterionType.PRIOR_AUTH,
        description="PA required",
        applies_to_drug="adalimumab",
    )
    return ExtractedPolicy(
        policy=policy,
        payer=payer,
        drugs=[drug],
        indications=[indication],
        criteria=[criterion],
        extractor_model="claude-sonnet-4-6",
        extraction_confidence=0.92,
    )


def test_extracted_policy_structure() -> None:
    ep = _make_extracted()
    assert ep.policy.policy_id == "pol_001"
    assert len(ep.drugs) == 1
    assert len(ep.criteria) == 1


def test_drug_by_name_found() -> None:
    ep = _make_extracted()
    assert ep.drug_by_name("adalimumab") is not None


def test_drug_by_name_not_found() -> None:
    ep = _make_extracted()
    assert ep.drug_by_name("pembrolizumab") is None


def test_criteria_for_drug() -> None:
    ep = _make_extracted()
    criteria = ep.criteria_for_drug("adalimumab")
    assert len(criteria) == 1
    assert criteria[0].criterion_id == "crit_001"


# ── Graph node / relationship models ──────────────────────────────────────────

def test_drug_node_properties_excludes_none() -> None:
    node = DrugNode(drug_id="adalimumab", name="adalimumab")
    props = node.properties()
    assert "generic_name" not in props
    assert props["drug_id"] == "adalimumab"


def test_policy_node_merge_key() -> None:
    node = PolicyNode(policy_id="pol_001", title="Test", payer_id="aetna")
    assert node.merge_key == {"policy_id": "pol_001"}


def test_covers_relationship_properties() -> None:
    rel = CoversRelationship(from_id="pol_001", to_id="adalimumab", coverage_status="covered", tier="Tier 3")
    props = rel.properties()
    assert props["coverage_status"] == "covered"
    assert "from_id" not in props


def test_requires_relationship_optional_fields() -> None:
    rel = RequiresRelationship(from_id="pol_001", to_id="crit_001", for_drug="adalimumab")
    props = rel.properties()
    assert props["for_drug"] == "adalimumab"
    assert "for_indication" not in props
