"""Tests for RxNorm and ICD-10 reference modules (Stage 7).

All HTTP calls are intercepted with httpx mock transport — no network required.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.reference.rxnorm import DrugClassResult, RxNormClient, RxNormResult
from src.reference.icd10 import ICD10Client, ICD10Result


# ── httpx mock helpers ────────────────────────────────────────────────────────

def _mock_response(body: Any, status: int = 200) -> httpx.Response:
    content = json.dumps(body).encode()
    return httpx.Response(status, content=content,
                          headers={"content-type": "application/json"})


def _transport_from_map(url_map: dict[str, Any]) -> httpx.MockTransport:
    """Build a MockTransport that dispatches by URL substring."""
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        for key, body in url_map.items():
            if key in url_str:
                return _mock_response(body)
        return _mock_response({}, status=404)

    return httpx.MockTransport(handler)


# ── RxNorm fixtures ───────────────────────────────────────────────────────────

_RXCUI_RESPONSE = {"idGroup": {"rxnormId": ["327361"]}}
_PROPERTY_RESPONSE = {
    "propConceptGroup": {
        "propConcept": [{"propName": "RxNorm Name", "propValue": "adalimumab"}]
    }
}
_RELATED_BN_RESPONSE = {
    "relatedGroup": {
        "conceptGroup": [
            {"tty": "BN", "conceptProperties": [
                {"name": "Humira"},
                {"name": "Hadlima"},
            ]}
        ]
    }
}
_RELATED_SY_RESPONSE = {"relatedGroup": {"conceptGroup": []}}
_CLASSES_RESPONSE = {
    "rxclassDrugInfoList": {
        "rxclassDrugInfo": [
            {"rxclassMinConceptItem": {
                "classId": "N0000175967",
                "className": "Tumor Necrosis Factor Inhibitor",
                "classType": "EPC",
            }},
            {"rxclassMinConceptItem": {
                "classId": "N0000000224",
                "className": "Tumor Necrosis Factor Inhibitors",
                "classType": "MOA",
            }},
        ]
    }
}


def _rxnorm_client() -> RxNormClient:
    url_map = {
        "rxcui.json":           _RXCUI_RESPONSE,
        "property.json":        _PROPERTY_RESPONSE,
        "related.json?tty=BN":  _RELATED_BN_RESPONSE,
        "related.json?tty=SY":  _RELATED_SY_RESPONSE,
        "classes.json":         _CLASSES_RESPONSE,
    }
    client = RxNormClient()
    client._http = httpx.Client(transport=_transport_from_map(url_map))
    return client


# ── RxNormClient tests ────────────────────────────────────────────────────────

def test_rxnorm_normalize_returns_result() -> None:
    result = _rxnorm_client().normalize("adalimumab")
    assert result is not None
    assert isinstance(result, RxNormResult)


def test_rxnorm_normalize_cui() -> None:
    result = _rxnorm_client().normalize("adalimumab")
    assert result is not None
    assert result.rxnorm_cui == "327361"


def test_rxnorm_normalize_standard_name() -> None:
    result = _rxnorm_client().normalize("Humira")
    assert result is not None
    assert result.standard_name == "adalimumab"


def test_rxnorm_normalize_brand_names() -> None:
    result = _rxnorm_client().normalize("adalimumab")
    assert result is not None
    assert "Humira" in result.brand_names
    assert "Hadlima" in result.brand_names


def test_rxnorm_normalize_drug_class() -> None:
    result = _rxnorm_client().normalize("adalimumab")
    assert result is not None
    assert any("Tumor Necrosis Factor" in c for c in result.drug_classes)


def test_rxnorm_normalize_no_match_returns_none() -> None:
    url_map = {
        "rxcui.json":        {"idGroup": {}},
        "approximateTerm":   {"approximateGroup": {"candidate": []}},
    }
    client = RxNormClient()
    client._http = httpx.Client(transport=_transport_from_map(url_map))
    assert client.normalize("completelymadeupdrugname12345") is None


def test_rxnorm_lookup_by_cui() -> None:
    result = _rxnorm_client().lookup_by_cui("327361")
    assert result is not None
    assert result.rxnorm_cui == "327361"


def test_rxnorm_get_drug_classes_epc_moa_only() -> None:
    classes = _rxnorm_client().get_drug_classes("327361")
    types = {c.class_type for c in classes}
    assert "EPC" in types
    assert "MOA" in types


def test_rxnorm_search_approximate() -> None:
    approx_response = {
        "approximateGroup": {
            "candidate": [
                {"rxcui": "327361", "name": "adalimumab"},
                {"rxcui": "327361", "name": "adalimumab"},   # duplicate should be deduped
                {"rxcui": "999999", "name": "adalimumab biosimilar"},
            ]
        }
    }
    client = RxNormClient()
    client._http = httpx.Client(transport=_transport_from_map({"approximateTerm": approx_response}))
    results = client.search_approximate("adalimumb")  # deliberate typo
    cuis = [r["cui"] for r in results]
    assert cuis.count("327361") == 1    # deduped
    assert "999999" in cuis


def test_rxnorm_http_error_returns_none() -> None:
    def _err_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = RxNormClient()
    client._http = httpx.Client(transport=httpx.MockTransport(_err_handler))
    result = client.normalize("adalimumab")
    assert result is None


def test_rxnorm_fallback_to_approximate_when_exact_empty() -> None:
    """If exact lookup returns empty idGroup, should fall back to approximate."""
    url_map = {
        "rxcui.json": {"idGroup": {}},
        "approximateTerm": {"approximateGroup": {"candidate": [{"rxcui": "327361", "name": "adalimumab"}]}},
        "property.json":   _PROPERTY_RESPONSE,
        "related.json?tty=BN": _RELATED_BN_RESPONSE,
        "related.json?tty=SY": _RELATED_SY_RESPONSE,
        "classes.json":    _CLASSES_RESPONSE,
    }
    client = RxNormClient()
    client._http = httpx.Client(transport=_transport_from_map(url_map))
    result = client.normalize("adalimumb")
    assert result is not None
    assert result.rxnorm_cui == "327361"


# ── ICD-10 fixtures ───────────────────────────────────────────────────────────

# NLM Clinical Tables response format: [total, codes, extra, [[code, desc], ...]]
_ICD10_SEARCH_RESPONSE = [
    4,
    ["M05.70", "M05.79", "M06.00", "M06.09"],
    None,
    [
        ["M05.70", "Rheumatoid arthritis with rheumatoid factor, unspecified site"],
        ["M05.79", "Rheumatoid arthritis with rheumatoid factor, multiple sites"],
        ["M06.00", "Rheumatoid arthritis without rheumatoid factor, unspecified site"],
        ["M06.09", "Rheumatoid arthritis without rheumatoid factor, multiple sites"],
    ],
]

_ICD10_SINGLE_RESPONSE = [
    1,
    ["L40.0"],
    None,
    [["L40.0", "Psoriasis vulgaris"]],
]


def _icd10_client(response: Any = _ICD10_SEARCH_RESPONSE) -> ICD10Client:
    def handler(req: httpx.Request) -> httpx.Response:
        return _mock_response(response)

    client = ICD10Client()
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


# ── ICD10Client tests ─────────────────────────────────────────────────────────

def test_icd10_search_returns_results() -> None:
    results = _icd10_client().search("rheumatoid arthritis")
    assert len(results) == 4
    assert all(isinstance(r, ICD10Result) for r in results)


def test_icd10_search_codes() -> None:
    results = _icd10_client().search("rheumatoid arthritis")
    codes = [r.code for r in results]
    assert "M05.70" in codes
    assert "M05.79" in codes


def test_icd10_search_descriptions() -> None:
    results = _icd10_client().search("rheumatoid arthritis")
    descs = [r.description for r in results]
    assert any("Rheumatoid arthritis" in d for d in descs)


def test_icd10_search_empty_returns_empty() -> None:
    results = _icd10_client(response=[0, [], None, []]).search("xyznonexistent")
    assert results == []


def test_icd10_search_malformed_returns_empty() -> None:
    results = _icd10_client(response=None).search("anything")
    assert results == []


def test_icd10_lookup_exact_match() -> None:
    result = _icd10_client(_ICD10_SINGLE_RESPONSE).lookup("L40.0")
    assert result is not None
    assert result.code == "L40.0"
    assert "Psoriasis" in result.description


def test_icd10_lookup_not_found_returns_none() -> None:
    # Response returns a different code, so exact match fails
    result = _icd10_client(_ICD10_SINGLE_RESPONSE).lookup("Z99.9")
    assert result is None


def test_icd10_lookup_normalises_code_without_dot() -> None:
    # "L400" should be normalised to "L40.0"
    result = _icd10_client(_ICD10_SINGLE_RESPONSE).lookup("L400")
    assert result is not None
    assert result.code == "L40.0"


def test_icd10_validate_code_true() -> None:
    assert _icd10_client(_ICD10_SINGLE_RESPONSE).validate_code("L40.0") is True


def test_icd10_validate_code_false() -> None:
    assert _icd10_client(_ICD10_SINGLE_RESPONSE).validate_code("Z99.9") is False


def test_icd10_result_is_valid_format() -> None:
    r = ICD10Result(code="M05.79", description="RA")
    assert r.is_valid_format is True


def test_icd10_result_invalid_format() -> None:
    r = ICD10Result(code="not-a-code", description="garbage")
    assert r.is_valid_format is False


def test_icd10_search_bulk() -> None:
    client = _icd10_client(_ICD10_SEARCH_RESPONSE)
    result = client.search_bulk(["rheumatoid arthritis", "psoriasis"])
    assert "rheumatoid arthritis" in result
    assert "psoriasis" in result
    assert len(result["rheumatoid arthritis"]) > 0


def test_icd10_http_error_returns_empty() -> None:
    def _err(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = ICD10Client()
    client._http = httpx.Client(transport=httpx.MockTransport(_err))
    assert client.search("anything") == []


# ── Normalizer tests ──────────────────────────────────────────────────────────

def test_enrich_drugs_fills_missing_cui() -> None:
    from src.models.policy import Drug
    from src.ingestion.normalizer import enrich_drugs

    drug = Drug(drug_id="adalimumab", name="adalimumab")
    assert drug.rxnorm_cui is None

    mock_rxnorm = MagicMock()
    mock_rxnorm.normalize.return_value = RxNormResult(
        rxnorm_cui="327361",
        standard_name="adalimumab",
        brand_names=["Humira"],
        drug_classes=["TNF inhibitor"],
    )

    enrich_drugs([drug], mock_rxnorm)
    assert drug.rxnorm_cui == "327361"
    assert drug.drug_class == "TNF inhibitor"
    assert "Humira" in drug.brand_names


def test_enrich_drugs_skips_already_enriched() -> None:
    from src.models.policy import Drug
    from src.ingestion.normalizer import enrich_drugs

    drug = Drug(drug_id="adalimumab", name="adalimumab", rxnorm_cui="327361")
    mock_rxnorm = MagicMock()

    enrich_drugs([drug], mock_rxnorm)
    mock_rxnorm.normalize.assert_not_called()


def test_enrich_drugs_handles_no_match() -> None:
    from src.models.policy import Drug
    from src.ingestion.normalizer import enrich_drugs

    drug = Drug(drug_id="unknown_drug", name="made_up_drug")
    mock_rxnorm = MagicMock()
    mock_rxnorm.normalize.return_value = None

    enrich_drugs([drug], mock_rxnorm)
    assert drug.rxnorm_cui is None


def test_enrich_indications_fills_missing_codes() -> None:
    from src.models.policy import Indication
    from src.ingestion.normalizer import enrich_indications

    ind = Indication(indication_id="ra", name="Rheumatoid Arthritis")
    mock_icd10 = MagicMock()
    mock_icd10.search.return_value = [
        ICD10Result(code="M05.79", description="RA multiple sites"),
    ]

    enrich_indications([ind], mock_icd10)
    assert "M05.79" in ind.icd10_codes


def test_enrich_indications_skips_already_coded() -> None:
    from src.models.policy import Indication
    from src.ingestion.normalizer import enrich_indications

    ind = Indication(indication_id="ra", name="RA", icd10_codes=["M05.79"])
    mock_icd10 = MagicMock()

    enrich_indications([ind], mock_icd10)
    mock_icd10.search.assert_not_called()


def test_enrich_policy_calls_both(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from pathlib import Path
    from src.ingestion.vlm_extractor import _build_extracted_policy
    from src.ingestion.normalizer import enrich_policy

    raw = json.loads((Path(__file__).parent / "fixtures" / "sample_extraction_response.json").read_text())
    ep = _build_extracted_policy(raw, model="test")

    mock_rxnorm = MagicMock()
    mock_rxnorm.normalize.return_value = None
    mock_icd10 = MagicMock()
    mock_icd10.search.return_value = []

    enrich_policy(ep, mock_rxnorm, mock_icd10)
    # Drugs that already have CUIs should NOT trigger normalize calls
    drugs_without_cui = [d for d in ep.drugs if not d.rxnorm_cui]
    assert mock_rxnorm.normalize.call_count == len(drugs_without_cui)


# ── Live API smoke tests (skipped without network) ────────────────────────────

@pytest.mark.skipif(
    os.getenv("CI") == "true" or not os.getenv("RUN_LIVE_REFERENCE_TESTS"),
    reason="Set RUN_LIVE_REFERENCE_TESTS=1 to run live API tests",
)
def test_live_rxnorm_adalimumab() -> None:
    with RxNormClient() as client:
        result = client.normalize("adalimumab")
    assert result is not None
    assert result.rxnorm_cui


@pytest.mark.skipif(
    os.getenv("CI") == "true" or not os.getenv("RUN_LIVE_REFERENCE_TESTS"),
    reason="Set RUN_LIVE_REFERENCE_TESTS=1 to run live API tests",
)
def test_live_icd10_rheumatoid_arthritis() -> None:
    with ICD10Client() as client:
        results = client.search("rheumatoid arthritis seropositive")
    assert len(results) > 0
    assert any("M05" in r.code for r in results)
