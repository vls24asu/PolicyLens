"""VLM-based PDF extraction using Claude vision + tool use.

The extractor sends every page as a base64 PNG to the Anthropic API and
asks Claude to return structured JSON via a tool call.  For PDFs longer than
MAX_PAGES_PER_BATCH pages the extraction runs in batches, and the per-batch
results are merged by _merge_extractions().
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import anthropic

from src.ingestion.pdf_loader import PDFLoader, PageData
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

logger = logging.getLogger(__name__)

# ── Extraction tool schema ────────────────────────────────────────────────────
# This is the JSON schema Claude fills in via tool use.  It mirrors
# ExtractedPolicy but uses primitive types so it works as an API schema.

_EXTRACTION_TOOL: dict[str, Any] = {
    "name": "extract_policy",
    "description": (
        "Extract all structured information from a medical benefit drug policy document. "
        "Call this tool exactly once with the complete extraction result."
    ),
    "input_schema": {
        "type": "object",
        "required": ["payer", "policy", "drugs", "indications", "criteria", "coverage_facts"],
        "properties": {
            "payer": {
                "type": "object",
                "required": ["payer_id", "name"],
                "properties": {
                    "payer_id":  {"type": "string", "description": "Lowercase slug, e.g. 'aetna'"},
                    "name":      {"type": "string"},
                    "type":      {"type": "string", "enum": ["commercial", "medicare", "medicaid", "exchange"]},
                    "website":   {"type": "string"},
                },
            },
            "policy": {
                "type": "object",
                "required": ["policy_id", "title", "payer_id"],
                "properties": {
                    "policy_id":          {"type": "string", "description": "Unique ID; generate a slug from payer + title if not present"},
                    "title":              {"type": "string"},
                    "payer_id":           {"type": "string"},
                    "effective_date":     {"type": "string", "description": "ISO 8601 date, e.g. '2024-01-01'"},
                    "last_reviewed_date": {"type": "string"},
                    "version":            {"type": "string"},
                    "source_url":         {"type": "string"},
                },
            },
            "plans": {
                "type": "array",
                "description": "Insurance plans this policy applies to (may be empty if not stated)",
                "items": {
                    "type": "object",
                    "required": ["plan_id", "name", "payer_id"],
                    "properties": {
                        "plan_id":   {"type": "string"},
                        "name":      {"type": "string"},
                        "payer_id":  {"type": "string"},
                        "plan_type": {"type": "string", "enum": ["hmo", "ppo", "epo", "pos", "hdhp", "other"]},
                    },
                },
            },
            "drugs": {
                "type": "array",
                "description": "All drugs mentioned (covered or excluded)",
                "items": {
                    "type": "object",
                    "required": ["drug_id", "name"],
                    "properties": {
                        "drug_id":      {"type": "string", "description": "Lowercase slug of generic name"},
                        "name":         {"type": "string", "description": "Primary name as it appears in the document"},
                        "generic_name": {"type": "string"},
                        "brand_names":  {"type": "array", "items": {"type": "string"}},
                        "rxnorm_cui":   {"type": "string"},
                        "hcpcs_code":   {"type": "string", "description": "J-code or Q-code if present"},
                        "drug_class":   {"type": "string", "description": "Therapeutic class, e.g. 'TNF inhibitor'"},
                        "ndc_codes":    {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "indications": {
                "type": "array",
                "description": "Clinical indications / diagnoses covered by this policy",
                "items": {
                    "type": "object",
                    "required": ["indication_id", "name"],
                    "properties": {
                        "indication_id": {"type": "string", "description": "Lowercase slug"},
                        "name":          {"type": "string"},
                        "description":   {"type": "string"},
                        "icd10_codes":   {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "criteria": {
                "type": "array",
                "description": "All coverage criteria: prior auth, step therapy, quantity limits, etc.",
                "items": {
                    "type": "object",
                    "required": ["criterion_id", "type", "description"],
                    "properties": {
                        "criterion_id":         {"type": "string"},
                        "type":                 {"type": "string", "enum": [
                            "prior_auth", "step_therapy", "quantity_limit",
                            "age_requirement", "lab_requirement",
                            "diagnosis_requirement", "prescriber_requirement", "other",
                        ]},
                        "description":          {"type": "string"},
                        "required_value":       {"type": "string"},
                        "applies_to_drug":      {"type": "string", "description": "drug_id"},
                        "applies_to_indication":{"type": "string", "description": "indication_id"},
                        "sequence":             {"type": "integer", "description": "Step order (1 = first-line)"},
                    },
                },
            },
            "coverage_facts": {
                "type": "array",
                "description": "Coverage status of each drug under this policy",
                "items": {
                    "type": "object",
                    "required": ["policy_id", "drug_id", "coverage_status"],
                    "properties": {
                        "policy_id":       {"type": "string"},
                        "drug_id":         {"type": "string"},
                        "coverage_status": {"type": "string", "enum": [
                            "covered", "not_covered", "covered_with_restrictions",
                            "investigational", "excluded",
                        ]},
                        "tier":  {"type": "string"},
                        "notes": {"type": "string"},
                    },
                },
            },
            "extraction_confidence": {
                "type": "number",
                "description": "Your confidence in the extraction accuracy (0.0–1.0)",
            },
            "extraction_warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Non-fatal issues: blurry pages, ambiguous text, missing sections, etc.",
            },
        },
    },
}

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a medical policy analyst specialising in health insurance drug coverage.
You will receive one or more pages from a Medical Benefit Drug Policy PDF as images.

Your task is to call the `extract_policy` tool with ALL structured information you
can identify in the document. Be thorough and precise.

Guidelines:
- Extract EVERY drug mentioned, whether covered or excluded.
- For each drug, capture the generic name, brand names, HCPCS/J-code, and
  RxNorm CUI if visible.
- Capture ALL prior authorisation (PA) criteria, step-therapy requirements,
  quantity limits, age requirements, and lab requirements as separate Criterion
  objects. Link each criterion to its drug_id and/or indication_id.
- For step therapy, set the `sequence` field (1 = first agent to try).
- Capture ICD-10 codes exactly as printed (e.g. "M05.79", "C34.10-C34.12").
- If a field is not present in the document, omit it rather than guessing.
- Generate stable slug IDs: payer_id = lowercase payer name with underscores;
  drug_id = lowercase generic name with underscores; indication_id = lowercase
  indication name with underscores.
- Set `extraction_confidence` to reflect how complete and legible the document is.
- Add entries to `extraction_warnings` for any blurry pages, tables that were
  hard to parse, or sections that seem incomplete.
"""

_USER_PROMPT_TEMPLATE = """\
Below are pages {start_page}–{end_page} of the policy document "{filename}".
{continuation}
Please call the `extract_policy` tool with everything you can extract from these pages.
"""


# ── VLMExtractor ──────────────────────────────────────────────────────────────

class VLMExtractor:
    """Extract structured policy data from a PDF using Claude vision + tool use.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` (or ``anthropic.AsyncAnthropic``) client.
    model:
        Claude model ID to use for extraction.
    dpi:
        Render resolution for PDF pages.  150 is a good default.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
        dpi: int = 150,
    ) -> None:
        self._client = client
        self._model = model
        self._loader = PDFLoader(dpi=dpi)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, pdf_path: str | Path) -> ExtractedPolicy:
        """Extract a policy from a PDF file.

        Converts each page to a PNG image, sends batches to Claude with the
        extraction tool, and merges the results into a validated ExtractedPolicy.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file.

        Returns
        -------
        ExtractedPolicy
            Fully validated Pydantic model.
        """
        path = Path(pdf_path)
        pdf_doc = self._loader.load(path)
        logger.info("Extracting: %s (%d pages)", path.name, pdf_doc.page_count)

        batches = pdf_doc.batches()
        raw_extractions: list[dict[str, Any]] = []

        for batch_idx, batch in enumerate(batches):
            logger.info(
                "  Batch %d/%d  pages %d–%d",
                batch_idx + 1, len(batches),
                batch[0].page_number, batch[-1].page_number,
            )
            raw = self._extract_batch(
                pages=batch,
                filename=path.name,
                is_continuation=batch_idx > 0,
            )
            raw_extractions.append(raw)

        merged = _merge_extractions(raw_extractions)

        # Attach document hash from the loader
        if "policy" in merged:
            merged["policy"]["document_hash"] = pdf_doc.document_hash
            merged["policy"]["raw_pdf_path"] = str(path)

        return _build_extracted_policy(merged, model=self._model)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_batch(
        self,
        pages: list[PageData],
        filename: str,
        is_continuation: bool,
    ) -> dict[str, Any]:
        """Send one batch of pages to Claude and return the raw tool input dict."""
        content: list[dict[str, Any]] = []

        # Attach each page as an image block
        for page in pages:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": page.image_base64,
                },
            })
            # Inline the extracted text as a hint (helps with tables)
            if page.text:
                content.append({
                    "type": "text",
                    "text": f"[Page {page.page_number} extracted text]\n{page.text[:2000]}",
                })

        # Append the instruction
        content.append({
            "type": "text",
            "text": _USER_PROMPT_TEMPLATE.format(
                start_page=pages[0].page_number,
                end_page=pages[-1].page_number,
                filename=filename,
                continuation=(
                    "This is a continuation — merge your findings with earlier pages. "
                    if is_continuation else ""
                ),
            ),
        })

        response = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=_SYSTEM_PROMPT,
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "any"},  # force a tool call
            messages=[{"role": "user", "content": content}],
        )

        # Find the tool use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_policy":
                return dict(block.input)  # type: ignore[arg-type]

        raise RuntimeError(
            f"Claude did not return a tool call for batch pages "
            f"{pages[0].page_number}–{pages[-1].page_number}"
        )


# ── Merge + build helpers ─────────────────────────────────────────────────────

def _merge_extractions(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple per-batch raw extractions into one dict.

    Strategy: first batch is the base; subsequent batches append unique items
    (deduped by ID) and accumulate warnings.
    """
    if not batches:
        return {}
    if len(batches) == 1:
        return batches[0]

    base = dict(batches[0])

    def _seen_ids(lst: list[dict[str, Any]], id_key: str) -> set[str]:
        return {item[id_key] for item in lst if id_key in item}

    list_fields = [
        ("plans",          "plan_id"),
        ("drugs",          "drug_id"),
        ("indications",    "indication_id"),
        ("criteria",       "criterion_id"),
        ("coverage_facts", None),   # no unique ID — append all and dedupe on (policy_id, drug_id)
    ]

    for batch in batches[1:]:
        for field_name, id_key in list_fields:
            existing: list[dict[str, Any]] = base.setdefault(field_name, [])
            incoming: list[dict[str, Any]] = batch.get(field_name, [])
            if id_key:
                seen = _seen_ids(existing, id_key)
                for item in incoming:
                    if item.get(id_key) not in seen:
                        existing.append(item)
                        seen.add(item[id_key])
            else:
                # coverage_facts — dedupe on (policy_id, drug_id)
                seen_pairs = {
                    (f.get("policy_id"), f.get("drug_id")) for f in existing
                }
                for item in incoming:
                    pair = (item.get("policy_id"), item.get("drug_id"))
                    if pair not in seen_pairs:
                        existing.append(item)
                        seen_pairs.add(pair)

        # Merge warnings
        base.setdefault("extraction_warnings", [])
        base["extraction_warnings"].extend(batch.get("extraction_warnings", []))

        # Average confidence scores
        if "extraction_confidence" in batch:
            base["extraction_confidence"] = (
                (base.get("extraction_confidence") or 1.0) + batch["extraction_confidence"]
            ) / 2

    return base


def _build_extracted_policy(raw: dict[str, Any], model: str) -> ExtractedPolicy:
    """Convert the raw dict from Claude into a validated ExtractedPolicy."""

    def _slug(text: str) -> str:
        return text.strip().lower().replace(" ", "_").replace("-", "_")[:64]

    # ── Payer ──────────────────────────────────────────────────────────────
    raw_payer = raw.get("payer", {})
    payer = Payer(
        payer_id=raw_payer.get("payer_id") or _slug(raw_payer.get("name", "unknown")),
        name=raw_payer.get("name", "Unknown"),
        type=_safe_enum(PayerType, raw_payer.get("type"), PayerType.COMMERCIAL),
        website=raw_payer.get("website"),
    )

    # ── Policy ─────────────────────────────────────────────────────────────
    raw_policy = raw.get("policy", {})
    policy_id = raw_policy.get("policy_id") or _make_policy_id(payer.payer_id, raw_policy.get("title", ""))
    policy = Policy(
        policy_id=policy_id,
        title=raw_policy.get("title", "Untitled Policy"),
        payer_id=payer.payer_id,
        effective_date=_parse_date(raw_policy.get("effective_date")),
        last_reviewed_date=_parse_date(raw_policy.get("last_reviewed_date")),
        version=raw_policy.get("version"),
        source_url=raw_policy.get("source_url"),
        document_hash=raw_policy.get("document_hash"),
        raw_pdf_path=raw_policy.get("raw_pdf_path"),
    )

    # ── Plans ──────────────────────────────────────────────────────────────
    plans = [
        Plan(
            plan_id=p.get("plan_id") or _slug(p.get("name", str(uuid.uuid4())[:8])),
            name=p.get("name", ""),
            payer_id=payer.payer_id,
            plan_type=_safe_enum(PlanType, p.get("plan_type"), PlanType.OTHER),
        )
        for p in raw.get("plans", [])
    ]

    # ── Drugs ──────────────────────────────────────────────────────────────
    drugs = [
        Drug(
            drug_id=d.get("drug_id") or _slug(d.get("name", str(uuid.uuid4())[:8])),
            name=d.get("name", ""),
            generic_name=d.get("generic_name"),
            brand_names=d.get("brand_names", []),
            rxnorm_cui=d.get("rxnorm_cui"),
            hcpcs_code=d.get("hcpcs_code"),
            drug_class=d.get("drug_class"),
            ndc_codes=d.get("ndc_codes", []),
        )
        for d in raw.get("drugs", [])
    ]

    # ── Indications ────────────────────────────────────────────────────────
    indications = [
        Indication(
            indication_id=i.get("indication_id") or _slug(i.get("name", str(uuid.uuid4())[:8])),
            name=i.get("name", ""),
            description=i.get("description"),
            icd10_codes=i.get("icd10_codes", []),
        )
        for i in raw.get("indications", [])
    ]

    # ── Criteria ───────────────────────────────────────────────────────────
    criteria = [
        Criterion(
            criterion_id=c.get("criterion_id") or str(uuid.uuid4())[:8],
            type=_safe_enum(CriterionType, c.get("type"), CriterionType.OTHER),
            description=c.get("description", ""),
            required_value=c.get("required_value"),
            applies_to_drug=c.get("applies_to_drug"),
            applies_to_indication=c.get("applies_to_indication"),
            sequence=c.get("sequence"),
        )
        for c in raw.get("criteria", [])
    ]

    # ── Coverage facts ─────────────────────────────────────────────────────
    coverage_facts = [
        CoverageFact(
            policy_id=policy_id,
            drug_id=f.get("drug_id", ""),
            coverage_status=_safe_enum(
                CoverageStatus, f.get("coverage_status"),
                CoverageStatus.COVERED_WITH_RESTRICTIONS,
            ),
            tier=f.get("tier"),
            notes=f.get("notes"),
        )
        for f in raw.get("coverage_facts", [])
    ]

    return ExtractedPolicy(
        policy=policy,
        payer=payer,
        plans=plans,
        drugs=drugs,
        indications=indications,
        criteria=criteria,
        coverage_facts=coverage_facts,
        extractor_model=model,
        extraction_confidence=raw.get("extraction_confidence"),
        extraction_warnings=raw.get("extraction_warnings", []),
    )


# ── Tiny utilities ────────────────────────────────────────────────────────────

def _make_policy_id(payer_id: str, title: str) -> str:
    slug = (payer_id + "_" + title).lower().replace(" ", "_")[:48]
    suffix = hashlib.sha256(slug.encode()).hexdigest()[:8]
    return f"{slug}_{suffix}"


def _parse_date(value: Any) -> Any:
    if not value:
        return None
    from datetime import date
    if isinstance(value, date):
        return value
    try:
        from dateutil.parser import parse
        return parse(str(value)).date()
    except Exception:
        return None


_E = type[Any]


def _safe_enum(enum_cls: _E, value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        return default
