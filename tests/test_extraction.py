"""Tests for PDF loading and VLM extraction (Stage 4).

Unit tests run without any API keys or real PDFs.
Live tests are skipped unless ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.vlm_extractor import (
    VLMExtractor,
    _build_extracted_policy,
    _merge_extractions,
    _make_policy_id,
)
from src.models.policy import (
    CoverageStatus,
    CriterionType,
    ExtractedPolicy,
    PayerType,
    PlanType,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FIXTURE = FIXTURES_DIR / "sample_extraction_response.json"


# ── Fixture helpers ───────────────────────────────────────────────────────────

@pytest.fixture()
def sample_raw() -> dict:
    return json.loads(SAMPLE_FIXTURE.read_text())


@pytest.fixture()
def sample_extracted(sample_raw: dict) -> ExtractedPolicy:
    return _build_extracted_policy(sample_raw, model="claude-sonnet-4-6")


# ── _build_extracted_policy unit tests ───────────────────────────────────────

def test_build_payer(sample_extracted: ExtractedPolicy) -> None:
    assert sample_extracted.payer.payer_id == "aetna"
    assert sample_extracted.payer.name == "Aetna"
    assert sample_extracted.payer.type == PayerType.COMMERCIAL


def test_build_policy_metadata(sample_extracted: ExtractedPolicy) -> None:
    pol = sample_extracted.policy
    assert pol.policy_id == "aetna_adalimumab_pa_policy_2024"
    assert pol.version == "4.2"
    assert pol.payer_id == "aetna"
    assert pol.effective_date is not None
    assert pol.effective_date.year == 2024


def test_build_plans(sample_extracted: ExtractedPolicy) -> None:
    assert len(sample_extracted.plans) == 2
    plan_types = {p.plan_type for p in sample_extracted.plans}
    assert PlanType.PPO in plan_types
    assert PlanType.HMO in plan_types


def test_build_drugs(sample_extracted: ExtractedPolicy) -> None:
    assert len(sample_extracted.drugs) == 2
    ada = sample_extracted.drug_by_name("adalimumab")
    assert ada is not None
    assert ada.hcpcs_code == "J0135"
    assert ada.rxnorm_cui == "327361"
    assert "Humira" in ada.brand_names


def test_build_indications(sample_extracted: ExtractedPolicy) -> None:
    assert len(sample_extracted.indications) == 3
    ids = {i.indication_id for i in sample_extracted.indications}
    assert "rheumatoid_arthritis" in ids
    assert "plaque_psoriasis" in ids


def test_build_indications_icd10(sample_extracted: ExtractedPolicy) -> None:
    ra = next(i for i in sample_extracted.indications if i.indication_id == "rheumatoid_arthritis")
    assert "M05.79" in ra.icd10_codes


def test_build_criteria_count(sample_extracted: ExtractedPolicy) -> None:
    assert len(sample_extracted.criteria) == 5


def test_build_criteria_types(sample_extracted: ExtractedPolicy) -> None:
    types = {c.type for c in sample_extracted.criteria}
    assert CriterionType.PRIOR_AUTH in types
    assert CriterionType.STEP_THERAPY in types
    assert CriterionType.QUANTITY_LIMIT in types
    assert CriterionType.AGE_REQUIREMENT in types
    assert CriterionType.LAB_REQUIREMENT in types


def test_build_step_therapy_sequence(sample_extracted: ExtractedPolicy) -> None:
    step = next(c for c in sample_extracted.criteria if c.type == CriterionType.STEP_THERAPY)
    assert step.sequence == 1
    assert step.applies_to_drug == "adalimumab"
    assert step.applies_to_indication == "rheumatoid_arthritis"


def test_build_coverage_facts(sample_extracted: ExtractedPolicy) -> None:
    assert len(sample_extracted.coverage_facts) == 1
    fact = sample_extracted.coverage_facts[0]
    assert fact.drug_id == "adalimumab"
    assert fact.coverage_status == CoverageStatus.COVERED_WITH_RESTRICTIONS
    assert fact.tier == "Specialty Tier"


def test_build_provenance(sample_extracted: ExtractedPolicy) -> None:
    assert sample_extracted.extractor_model == "claude-sonnet-4-6"
    assert sample_extracted.extraction_confidence == pytest.approx(0.94)
    assert len(sample_extracted.extraction_warnings) == 2


def test_criteria_for_drug_helper(sample_extracted: ExtractedPolicy) -> None:
    criteria = sample_extracted.criteria_for_drug("adalimumab")
    assert len(criteria) == 5


# ── Graceful handling of missing / malformed fields ──────────────────────────

def test_build_missing_payer_name() -> None:
    raw = {"payer": {}, "policy": {"title": "Test", "payer_id": "x"}, "drugs": [], "indications": [], "criteria": [], "coverage_facts": []}
    ep = _build_extracted_policy(raw, model="test")
    assert ep.payer.name == "Unknown"
    assert ep.payer.payer_id == "unknown"


def test_build_unknown_criterion_type() -> None:
    raw = {
        "payer": {"payer_id": "aetna", "name": "Aetna"},
        "policy": {"policy_id": "p1", "title": "T", "payer_id": "aetna"},
        "drugs": [],
        "indications": [],
        "coverage_facts": [],
        "criteria": [{"criterion_id": "c1", "type": "TOTALLY_UNKNOWN", "description": "X"}],
    }
    ep = _build_extracted_policy(raw, model="test")
    assert ep.criteria[0].type == CriterionType.OTHER


def test_build_invalid_date_skipped() -> None:
    raw = {
        "payer": {"payer_id": "aetna", "name": "Aetna"},
        "policy": {"policy_id": "p1", "title": "T", "payer_id": "aetna", "effective_date": "not-a-date"},
        "drugs": [], "indications": [], "criteria": [], "coverage_facts": [],
    }
    ep = _build_extracted_policy(raw, model="test")
    assert ep.policy.effective_date is None


def test_policy_id_auto_generated_when_missing() -> None:
    raw = {
        "payer": {"payer_id": "cigna", "name": "Cigna"},
        "policy": {"title": "GLP-1 Coverage Policy", "payer_id": "cigna"},
        "drugs": [], "indications": [], "criteria": [], "coverage_facts": [],
    }
    ep = _build_extracted_policy(raw, model="test")
    assert ep.policy.policy_id.startswith("cigna_")


def test_make_policy_id_deterministic() -> None:
    id1 = _make_policy_id("aetna", "Biologic PA Policy")
    id2 = _make_policy_id("aetna", "Biologic PA Policy")
    assert id1 == id2


# ── _merge_extractions unit tests ─────────────────────────────────────────────

def test_merge_single_batch(sample_raw: dict) -> None:
    merged = _merge_extractions([sample_raw])
    assert merged is sample_raw


def test_merge_two_batches_dedupes_drugs(sample_raw: dict) -> None:
    batch2 = {
        "payer": sample_raw["payer"],
        "policy": sample_raw["policy"],
        "drugs": [
            # duplicate
            {"drug_id": "adalimumab", "name": "adalimumab"},
            # new drug
            {"drug_id": "etanercept", "name": "etanercept", "brand_names": ["Enbrel"]},
        ],
        "indications": [],
        "criteria": [],
        "coverage_facts": [],
        "extraction_warnings": ["Page 12 was blurry."],
    }
    merged = _merge_extractions([sample_raw, batch2])
    drug_ids = [d["drug_id"] for d in merged["drugs"]]
    assert drug_ids.count("adalimumab") == 1
    assert "etanercept" in drug_ids


def test_merge_accumulates_warnings(sample_raw: dict) -> None:
    batch2 = {**sample_raw, "extraction_warnings": ["Extra warning from batch 2."]}
    merged = _merge_extractions([sample_raw, batch2])
    assert any("batch 2" in w for w in merged["extraction_warnings"])


def test_merge_empty_list() -> None:
    assert _merge_extractions([]) == {}


# ── VLMExtractor with mocked Anthropic client ────────────────────────────────

def _make_mock_client(fixture_path: Path) -> MagicMock:
    """Return a mock Anthropic client that returns the recorded fixture."""
    raw = json.loads(fixture_path.read_text())

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "extract_policy"
    tool_use_block.input = raw

    response = MagicMock()
    response.content = [tool_use_block]

    client = MagicMock()
    client.messages.create.return_value = response
    return client


def _make_minimal_pdf(tmp_path: Path) -> Path:
    """Create a tiny 1-page PDF for loader smoke-testing."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Adalimumab (Humira) Prior Authorization Policy\nAetna Commercial Plans\nEffective Date: January 1, 2024")
        pdf_path = tmp_path / "sample_policy.pdf"
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    except ImportError:
        pytest.skip("PyMuPDF not installed")


def test_extractor_calls_api_once(tmp_path: Path) -> None:
    pdf_path = _make_minimal_pdf(tmp_path)
    mock_client = _make_mock_client(SAMPLE_FIXTURE)
    extractor = VLMExtractor(client=mock_client, model="claude-sonnet-4-6")
    result = extractor.extract(pdf_path)
    mock_client.messages.create.assert_called_once()
    assert isinstance(result, ExtractedPolicy)


def test_extractor_returns_valid_model(tmp_path: Path) -> None:
    pdf_path = _make_minimal_pdf(tmp_path)
    mock_client = _make_mock_client(SAMPLE_FIXTURE)
    extractor = VLMExtractor(client=mock_client)
    result = extractor.extract(pdf_path)
    assert result.policy.payer_id == "aetna"
    assert len(result.drugs) == 2
    assert len(result.criteria) == 5


def test_extractor_attaches_document_hash(tmp_path: Path) -> None:
    pdf_path = _make_minimal_pdf(tmp_path)
    mock_client = _make_mock_client(SAMPLE_FIXTURE)
    extractor = VLMExtractor(client=mock_client)
    result = extractor.extract(pdf_path)
    assert result.policy.document_hash is not None
    assert len(result.policy.document_hash) == 64  # SHA-256 hex


def test_extractor_raises_on_no_tool_call(tmp_path: Path) -> None:
    pdf_path = _make_minimal_pdf(tmp_path)
    text_block = MagicMock()
    text_block.type = "text"
    response = MagicMock()
    response.content = [text_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = response

    extractor = VLMExtractor(client=mock_client)
    with pytest.raises(RuntimeError, match="did not return a tool call"):
        extractor.extract(pdf_path)


# ── PDFLoader unit tests ──────────────────────────────────────────────────────

def test_pdf_loader_produces_page_data(tmp_path: Path) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")

    from src.ingestion.pdf_loader import PDFLoader
    pdf_path = _make_minimal_pdf(tmp_path)
    loader = PDFLoader(dpi=72)
    doc = loader.load(pdf_path)
    assert doc.page_count == 1
    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.page_number == 1
    assert len(page.image_bytes) > 0
    # Verify base64 round-trips
    assert base64.b64decode(page.image_base64) == page.image_bytes


def test_pdf_loader_raises_on_missing_file() -> None:
    from src.ingestion.pdf_loader import PDFLoader
    loader = PDFLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/path/policy.pdf")


def test_pdf_loader_batches(tmp_path: Path) -> None:
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")

    from src.ingestion.pdf_loader import PDFLoader
    # Build a 3-page PDF
    doc = fitz.open()
    for _ in range(3):
        p = doc.new_page()
        p.insert_text((72, 72), "Page content")
    pdf_path = tmp_path / "three_page.pdf"
    doc.save(str(pdf_path))
    doc.close()

    loader = PDFLoader(dpi=72)
    pdf_doc = loader.load(pdf_path)
    batches = pdf_doc.batches(batch_size=2)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


# ── Live integration test (skipped without API key) ───────────────────────────

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY")
def test_live_extraction_smoke(tmp_path: Path) -> None:
    import anthropic as _anthropic
    from src.config import settings

    pdf_path = _make_minimal_pdf(tmp_path)
    client = _anthropic.Anthropic(api_key=settings.anthropic_api_key)
    extractor = VLMExtractor(client=client, model=settings.vlm_model)
    result = extractor.extract(pdf_path)
    assert isinstance(result, ExtractedPolicy)
    assert result.policy.payer_id != ""
