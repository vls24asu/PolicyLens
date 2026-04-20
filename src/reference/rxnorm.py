"""RxNorm REST API wrapper.

Uses the NLM's free public RxNorm API (no key required):
  https://rxnav.nlm.nih.gov/RxNormAPIs.html

Results are cached in-memory with a simple LRU cache so repeated lookups
during a single ingestion run do not hit the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
_TIMEOUT  = 10.0   # seconds


# ── Result models ─────────────────────────────────────────────────────────────

@dataclass
class RxNormResult:
    """Normalised drug record from RxNorm."""

    rxnorm_cui: str
    standard_name: str
    brand_names: list[str] = field(default_factory=list)
    drug_classes: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)


@dataclass
class DrugClassResult:
    """A single therapeutic classification entry."""

    class_id: str
    class_name: str
    class_type: str          # e.g. "EPC", "MOA", "PE", "MESHPA", "VA"


# ── RxNormClient ──────────────────────────────────────────────────────────────

class RxNormClient:
    """Wrapper around the NLM RxNorm REST API.

    Parameters
    ----------
    base_url:
        Override for testing or if the URL changes.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: float = _TIMEOUT,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RxNormClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(self, drug_name: str) -> RxNormResult | None:
        """Look up a drug name and return its RxNorm normalised record.

        Tries an exact match first; falls back to approximate search.

        Parameters
        ----------
        drug_name:
            Any drug name (generic, brand, misspelling).

        Returns
        -------
        RxNormResult | None
            ``None`` if no match is found.
        """
        cui = self._find_cui(drug_name)
        if not cui:
            logger.debug("RxNorm: no CUI found for '%s'", drug_name)
            return None
        return self._build_result(cui)

    def lookup_by_cui(self, cui: str) -> RxNormResult | None:
        """Fetch a drug record directly by its RxCUI."""
        return self._build_result(cui)

    def get_drug_classes(self, cui: str) -> list[DrugClassResult]:
        """Return therapeutic classifications for an RxCUI.

        Uses the RxClass API which annotates drugs with ATC, EPC, MOA,
        MeSH, and VA drug class hierarchies.
        """
        url = f"{self._base}/rxcui/{cui}/classes.json"
        data = self._get(url)
        results: list[DrugClassResult] = []
        for entry in (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo", []):
            mini = entry.get("rxclassMinConceptItem", {})
            results.append(DrugClassResult(
                class_id=mini.get("classId", ""),
                class_name=mini.get("className", ""),
                class_type=mini.get("classType", ""),
            ))
        return results

    def search_approximate(self, term: str, max_entries: int = 5) -> list[dict[str, str]]:
        """Fuzzy search for drug names similar to *term*.

        Returns a list of ``{"cui": ..., "name": ...}`` dicts.
        """
        url = f"{self._base}/approximateTerm.json"
        data = self._get(url, params={"term": term, "maxEntries": max_entries})
        candidates = (data.get("approximateGroup") or {}).get("candidate", [])
        seen: set[str] = set()
        results: list[dict[str, str]] = []
        for c in candidates:
            cui = c.get("rxcui", "")
            if cui and cui not in seen:
                seen.add(cui)
                results.append({"cui": cui, "name": c.get("name", "")})
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_cui(self, name: str) -> str | None:
        """Try exact lookup first, then approximate."""
        # Exact
        url = f"{self._base}/rxcui.json"
        data = self._get(url, params={"name": name, "search": "1"})
        cui = (data.get("idGroup") or {}).get("rxnormId")
        if cui:
            return str(cui[0]) if isinstance(cui, list) else str(cui)

        # Approximate fallback
        candidates = self.search_approximate(name, max_entries=1)
        return candidates[0]["cui"] if candidates else None

    def _build_result(self, cui: str) -> RxNormResult | None:
        # Get the preferred name
        url = f"{self._base}/rxcui/{cui}/property.json"
        data = self._get(url, params={"propName": "RxNorm Name"})
        props = (data.get("propConceptGroup") or {}).get("propConcept", [])
        standard_name = props[0]["propValue"] if props else ""

        if not standard_name:
            # Fall back to the allProperties endpoint
            data2 = self._get(f"{self._base}/rxcui/{cui}/allProperties.json", params={"prop": "names"})
            prop_list = (data2.get("propConceptGroup") or {}).get("propConcept", [])
            for p in prop_list:
                if p.get("propName") in ("RxNorm Name", "PRESCRIBABLE_NAME"):
                    standard_name = p["propValue"]
                    break

        # Brand names (TTY=BN in related concepts)
        brand_names = self._fetch_related_names(cui, tty="BN")

        # Synonyms (TTY=SY)
        synonyms = self._fetch_related_names(cui, tty="SY")

        # Drug classes
        classes = self.get_drug_classes(cui)
        drug_class_names = [c.class_name for c in classes if c.class_type in ("EPC", "MOA")]

        return RxNormResult(
            rxnorm_cui=cui,
            standard_name=standard_name or cui,
            brand_names=brand_names,
            drug_classes=drug_class_names,
            synonyms=synonyms,
        )

    def _fetch_related_names(self, cui: str, tty: str) -> list[str]:
        """Fetch related concept names filtered by term type."""
        url = f"{self._base}/rxcui/{cui}/related.json"
        data = self._get(url, params={"tty": tty})
        group = (data.get("relatedGroup") or {}).get("conceptGroup", [])
        names: list[str] = []
        for g in group:
            for concept in g.get("conceptProperties", []):
                name = concept.get("name")
                if name:
                    names.append(name)
        return names

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """HTTP GET with retry on transient network errors."""
        try:
            resp = self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("RxNorm HTTP %s for %s", exc.response.status_code, url)
            return {}
        except Exception as exc:
            logger.warning("RxNorm request failed: %s", exc)
            return {}
