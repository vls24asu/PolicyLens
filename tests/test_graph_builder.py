"""Tests for GraphBuilder (Stage 5).

All tests mock the Neo4jClient — no live database required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch
import pytest

from src.ingestion.graph_builder import (
    GraphBuilder,
    _criterion_props,
    _drug_props,
    _indication_props,
    _payer_props,
    _plan_props,
    _policy_props,
)
from src.ingestion.vlm_extractor import _build_extracted_policy
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

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def full_extracted() -> ExtractedPolicy:
    raw = json.loads((FIXTURES_DIR / "sample_extraction_response.json").read_text())
    return _build_extracted_policy(raw, model="claude-sonnet-4-6")


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    client.run_write = AsyncMock(return_value=[])
    client.run = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def builder(mock_client: MagicMock) -> GraphBuilder:
    return GraphBuilder(mock_client)


# ── Property converter unit tests ─────────────────────────────────────────────

def test_payer_props_excludes_none(full_extracted: ExtractedPolicy) -> None:
    props = _payer_props(full_extracted)
    assert "payer_id" in props
    assert "name" in props
    assert "type" in props
    # website present in fixture
    assert "website" in props


def test_policy_props_iso_date(full_extracted: ExtractedPolicy) -> None:
    props = _policy_props(full_extracted)
    assert props["effective_date"] == "2024-01-01"
    assert props["last_reviewed_date"] == "2023-11-15"


def test_drug_props_brand_names(full_extracted: ExtractedPolicy) -> None:
    ada = next(d for d in full_extracted.drugs if d.drug_id == "adalimumab")
    props = _drug_props(ada)
    assert props["hcpcs_code"] == "J0135"
    assert "Humira" in props["brand_names"]


def test_drug_props_skips_none_fields() -> None:
    drug = Drug(drug_id="testdrug", name="testdrug")
    props = _drug_props(drug)
    assert "rxnorm_cui" not in props
    assert "hcpcs_code" not in props


def test_indication_props_icd10(full_extracted: ExtractedPolicy) -> None:
    ra = next(i for i in full_extracted.indications if i.indication_id == "rheumatoid_arthritis")
    props = _indication_props(ra)
    assert "M05.79" in props["icd10_codes"]


def test_criterion_props_step_therapy(full_extracted: ExtractedPolicy) -> None:
    step = next(c for c in full_extracted.criteria if c.type == CriterionType.STEP_THERAPY)
    props = _criterion_props(step)
    assert props["type"] == "step_therapy"
    assert props["sequence"] == 1


def test_plan_props_includes_plan_type(full_extracted: ExtractedPolicy) -> None:
    plan = full_extracted.plans[0]
    props = _plan_props(plan, "aetna")
    assert props["payer_id"] == "aetna"
    assert props["plan_type"] in ("ppo", "hmo", "other")


# ── GraphBuilder.build — node upsert calls ────────────────────────────────────

@pytest.mark.asyncio
async def test_build_upserts_payer(builder: GraphBuilder, mock_client: MagicMock, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        mock_q.upsert_payer = AsyncMock()
        mock_q.upsert_plan = AsyncMock()
        mock_q.upsert_policy = AsyncMock()
        mock_q.upsert_drug = AsyncMock()
        mock_q.upsert_indication = AsyncMock()
        mock_q.upsert_criterion = AsyncMock()
        mock_q.upsert_excerpt = AsyncMock()
        _patch_rel_mocks(mock_q)
        await builder.build(full_extracted)
    mock_q.upsert_payer.assert_awaited_once()
    payer_args = mock_q.upsert_payer.call_args.args[1]
    assert payer_args["payer_id"] == "aetna"


@pytest.mark.asyncio
async def test_build_upserts_all_drugs(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    assert mock_q.upsert_drug.await_count == len(full_extracted.drugs)


@pytest.mark.asyncio
async def test_build_upserts_all_criteria(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    assert mock_q.upsert_criterion.await_count == len(full_extracted.criteria)


@pytest.mark.asyncio
async def test_build_upserts_all_plans(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    assert mock_q.upsert_plan.await_count == len(full_extracted.plans)


# ── GraphBuilder.build — relationship calls ───────────────────────────────────

@pytest.mark.asyncio
async def test_build_links_payer_to_plans(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    assert mock_q.link_payer_plan.await_count == len(full_extracted.plans)


@pytest.mark.asyncio
async def test_build_links_plans_to_policy(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    assert mock_q.link_plan_policy.await_count == len(full_extracted.plans)


@pytest.mark.asyncio
async def test_build_covers_drug(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    # fixture has 1 coverage fact for adalimumab (covered_with_restrictions)
    mock_q.link_policy_covers_drug.assert_awaited_once()
    call_args = mock_q.link_policy_covers_drug.call_args
    # signature: link_policy_covers_drug(client, policy_id, drug_id, coverage_status, tier)
    assert call_args.args[1] == "aetna_adalimumab_pa_policy_2024"
    assert call_args.args[2] == "adalimumab"


@pytest.mark.asyncio
async def test_build_excluded_drug_uses_excludes_link() -> None:
    ep = _minimal_extracted_with_exclusion()
    mock_client = MagicMock()
    mock_client.run_write = AsyncMock(return_value=[])
    b = GraphBuilder(mock_client)
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await b.build(ep)
    mock_q.link_policy_excludes_drug.assert_awaited_once()
    mock_q.link_policy_covers_drug.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_infers_drug_treats_indication(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    # fixture has criteria linking adalimumab → rheumatoid_arthritis and plaque_psoriasis
    assert mock_q.link_drug_treats.await_count >= 2


@pytest.mark.asyncio
async def test_build_requires_all_criteria(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    assert mock_q.link_policy_requires.await_count == len(full_extracted.criteria)


@pytest.mark.asyncio
async def test_build_no_excerpts_skips_cites(builder: GraphBuilder, full_extracted: ExtractedPolicy) -> None:
    assert len(full_extracted.excerpts) == 0
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await builder.build(full_extracted)
    mock_q.link_policy_cites.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_with_excerpts_links_cites() -> None:
    ep = _minimal_extracted_with_excerpt()
    mc = MagicMock()
    mc.run_write = AsyncMock(return_value=[])
    b = GraphBuilder(mc)
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await b.build(ep)
    mock_q.link_policy_cites.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_default_coverage_when_no_facts() -> None:
    """Drugs with no coverage_facts should be linked as covered_with_restrictions."""
    ep = _minimal_extracted_no_facts()
    mc = MagicMock()
    mc.run_write = AsyncMock(return_value=[])
    b = GraphBuilder(mc)
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        await b.build(ep)
    assert mock_q.link_policy_covers_drug.await_count == len(ep.drugs)


@pytest.mark.asyncio
async def test_mark_supersedes(builder: GraphBuilder) -> None:
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        mock_q.link_policy_supersedes = AsyncMock()
        await builder.mark_supersedes("pol_v2", "pol_v1")
    mock_q.link_policy_supersedes.assert_awaited_once_with(
        builder._client, "pol_v2", "pol_v1"
    )


# ── Warning for unknown drug_id in coverage_facts ─────────────────────────────

@pytest.mark.asyncio
async def test_build_warns_on_unknown_drug_in_fact(caplog: pytest.LogCaptureFixture) -> None:
    ep = _minimal_extracted_with_unknown_drug_fact()
    mc = MagicMock()
    mc.run_write = AsyncMock(return_value=[])
    b = GraphBuilder(mc)
    with patch("src.ingestion.graph_builder.queries") as mock_q:
        _patch_all_mocks(mock_q)
        import logging
        with caplog.at_level(logging.WARNING, logger="src.ingestion.graph_builder"):
            await b.build(ep)
    assert any("unknown drug_id" in r.message for r in caplog.records)


# ── Helper factories ──────────────────────────────────────────────────────────

def _base_extracted(**overrides: object) -> ExtractedPolicy:
    payer = Payer(payer_id="test_payer", name="Test Payer")
    policy = Policy(policy_id="pol_test", title="Test Policy", payer_id="test_payer")
    defaults = dict(payer=payer, policy=policy, drugs=[], indications=[], criteria=[], coverage_facts=[], excerpts=[])
    defaults.update(overrides)  # type: ignore[arg-type]
    return ExtractedPolicy(**defaults)  # type: ignore[arg-type]


def _minimal_extracted_with_exclusion() -> ExtractedPolicy:
    drug = Drug(drug_id="excluded_drug", name="excluded_drug")
    fact = CoverageFact(policy_id="pol_test", drug_id="excluded_drug", coverage_status=CoverageStatus.EXCLUDED)
    return _base_extracted(drugs=[drug], coverage_facts=[fact])


def _minimal_extracted_with_excerpt() -> ExtractedPolicy:
    exc = SourceExcerpt(excerpt_id="exc_001", policy_id="pol_test", text="PA required.", page_number=1)
    return _base_extracted(excerpts=[exc])


def _minimal_extracted_no_facts() -> ExtractedPolicy:
    drugs = [
        Drug(drug_id="drug_a", name="Drug A"),
        Drug(drug_id="drug_b", name="Drug B"),
    ]
    return _base_extracted(drugs=drugs, coverage_facts=[])


def _minimal_extracted_with_unknown_drug_fact() -> ExtractedPolicy:
    fact = CoverageFact(policy_id="pol_test", drug_id="ghost_drug", coverage_status=CoverageStatus.COVERED)
    return _base_extracted(coverage_facts=[fact])


def _patch_all_mocks(mock_q: MagicMock) -> None:
    """Patch every query function to an AsyncMock."""
    for name in [
        "upsert_payer", "upsert_plan", "upsert_policy", "upsert_drug",
        "upsert_indication", "upsert_criterion", "upsert_excerpt",
        "link_payer_plan", "link_plan_policy",
        "link_policy_covers_drug", "link_policy_excludes_drug",
        "link_policy_indication", "link_drug_treats",
        "link_policy_requires", "link_policy_cites",
        "link_policy_supersedes",
    ]:
        setattr(mock_q, name, AsyncMock())


def _patch_rel_mocks(mock_q: MagicMock) -> None:
    for name in [
        "link_payer_plan", "link_plan_policy",
        "link_policy_covers_drug", "link_policy_excludes_drug",
        "link_policy_indication", "link_drug_treats",
        "link_policy_requires", "link_policy_cites",
    ]:
        setattr(mock_q, name, AsyncMock())
