"""Reference Data MCP server.

Exposes RxNorm drug normalisation and ICD-10 code lookup as MCP tools,
letting Claude resolve drug names and diagnosis codes to authoritative identifiers.

Run standalone:
    python scripts/run_mcp_server.py reference_data
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── Lazy clients ──────────────────────────────────────────────────────────────

_rxnorm_client: Any = None
_icd10_client: Any = None


def _get_rxnorm() -> Any:
    global _rxnorm_client
    if _rxnorm_client is None:
        from src.reference.rxnorm import RxNormClient
        _rxnorm_client = RxNormClient()
    return _rxnorm_client


def _get_icd10() -> Any:
    global _icd10_client
    if _icd10_client is None:
        from src.reference.icd10 import ICD10Client
        _icd10_client = ICD10Client()
    return _icd10_client


# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "reference-data",
    instructions=(
        "Resolve drug names to RxNorm CUIs and look up ICD-10-CM diagnosis codes "
        "using authoritative NLM terminology services."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def normalize_drug_name(input: str) -> dict[str, Any]:
    """Normalise a drug name to its RxNorm standard form.

    Accepts generic names, brand names, or misspellings and returns the
    canonical RxNorm record including CUI, standard name, brand names,
    and therapeutic class.

    Args:
        input: Any drug name string (e.g. "Humira", "adalimumb", "pembrolizumab").

    Returns:
        Dict with rxnorm_cui, standard_name, brand_names, drug_classes, synonyms.
        Returns {error: ...} if no match is found.
    """
    rxnorm = _get_rxnorm()
    result = rxnorm.normalize(input)
    if not result:
        return {"error": f"No RxNorm record found for '{input}'."}
    return {
        "rxnorm_cui":   result.rxnorm_cui,
        "standard_name": result.standard_name,
        "brand_names":  result.brand_names,
        "drug_classes": result.drug_classes,
        "synonyms":     result.synonyms[:5],   # trim to avoid response bloat
    }


@mcp.tool()
def lookup_icd10(description: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search for ICD-10-CM codes matching a clinical description.

    Args:
        description: Clinical description, e.g. "non-small cell lung cancer" or "RA seropositive".
        max_results: Maximum number of codes to return (default 10).

    Returns:
        List of {code, description} sorted by relevance.
        Returns [{error: ...}] if no codes found.
    """
    icd10 = _get_icd10()
    results = icd10.search(description, max_results=max_results)
    if not results:
        return [{"error": f"No ICD-10 codes found for '{description}'."}]
    return [{"code": r.code, "description": r.description} for r in results]


@mcp.tool()
def get_drug_class(drug_name: str) -> dict[str, Any]:
    """Return the therapeutic classification for a drug.

    First normalises the drug name to get its RxCUI, then fetches EPC and MOA
    class information from the RxClass API.

    Args:
        drug_name: Generic or brand name of the drug.

    Returns:
        Dict with rxnorm_cui, standard_name, and classes list
        [{class_id, class_name, class_type}].
        Returns {error: ...} if the drug is not found.
    """
    rxnorm = _get_rxnorm()
    norm = rxnorm.normalize(drug_name)
    if not norm:
        return {"error": f"Drug '{drug_name}' not found in RxNorm."}

    classes = rxnorm.get_drug_classes(norm.rxnorm_cui)
    return {
        "rxnorm_cui":   norm.rxnorm_cui,
        "standard_name": norm.standard_name,
        "classes": [
            {
                "class_id":   c.class_id,
                "class_name": c.class_name,
                "class_type": c.class_type,
            }
            for c in classes
        ],
    }


if __name__ == "__main__":
    mcp.run()
