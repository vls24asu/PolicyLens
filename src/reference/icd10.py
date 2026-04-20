"""ICD-10-CM code lookup using the NLM Clinical Tables API.

Endpoint docs:
  https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html

No API key required; public / free service.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

_BASE_URL  = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
_TIMEOUT   = 10.0
_MAX_LIST  = 20

# Pattern for a valid ICD-10-CM code: letter + 2 digits, optional decimal + 1-4 chars
_ICD10_RE  = re.compile(r'^[A-Z]\d{2}(\.\w{1,4})?$', re.IGNORECASE)


# ── Result models ─────────────────────────────────────────────────────────────

@dataclass
class ICD10Result:
    """A single ICD-10-CM code + description."""

    code: str
    description: str

    @property
    def is_valid_format(self) -> bool:
        return bool(_ICD10_RE.match(self.code))


@dataclass
class ICD10LookupResult:
    """Full lookup response including hierarchy info."""

    code: str
    description: str
    category: str         # E.g. "Diseases of the musculoskeletal system"
    is_billable: bool


# ── ICD10Client ───────────────────────────────────────────────────────────────

class ICD10Client:
    """Lookup ICD-10-CM codes via the NLM Clinical Tables API.

    Parameters
    ----------
    base_url:
        Override for testing.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: float = _TIMEOUT,
    ) -> None:
        self._base = base_url
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ICD10Client":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        description: str,
        max_results: int = 10,
    ) -> list[ICD10Result]:
        """Search ICD-10-CM codes by description text.

        Parameters
        ----------
        description:
            Clinical description, e.g. "rheumatoid arthritis seropositive".
        max_results:
            Maximum number of results to return (max 500 per API docs).

        Returns
        -------
        list[ICD10Result]
            Matching codes ordered by relevance.
        """
        data = self._get(params={
            "sf":      "code,name",
            "terms":   description,
            "maxList": min(max_results, _MAX_LIST),
        })
        return self._parse_search_response(data)

    def lookup(self, code: str) -> ICD10Result | None:
        """Look up a single ICD-10 code.

        Parameters
        ----------
        code:
            Exact ICD-10-CM code, e.g. ``"M05.79"`` (dot optional).

        Returns
        -------
        ICD10Result | None
            ``None`` if the code is not found.
        """
        normalised = self._normalise_code(code)
        data = self._get(params={
            "sf":      "code,name",
            "terms":   normalised,
            "maxList": 5,
        })
        results = self._parse_search_response(data)
        for r in results:
            if r.code.upper() == normalised.upper():
                return r
        return None

    def validate_code(self, code: str) -> bool:
        """Return True if *code* exists in the ICD-10-CM database."""
        return self.lookup(code) is not None

    def search_bulk(self, descriptions: list[str]) -> dict[str, list[ICD10Result]]:
        """Look up multiple descriptions, returning a dict keyed by description."""
        return {desc: self.search(desc) for desc in descriptions}

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalise_code(code: str) -> str:
        """Normalise a code to uppercase with dot if missing (e.g. 'M0579' → 'M05.79')."""
        code = code.strip().upper()
        # Insert dot if missing and length suggests it
        if "." not in code and len(code) > 3:
            code = code[:3] + "." + code[3:]
        return code

    @staticmethod
    def _parse_search_response(data: Any) -> list[ICD10Result]:
        """Parse the NLM Clinical Tables API response format.

        The API returns a 4-element list:
          [total_count, codes_list, extra_data, display_strings]
        where display_strings is a list of [code, description] pairs.
        """
        if not isinstance(data, list) or len(data) < 4:
            return []
        display = data[3]
        if not display:
            return []
        results: list[ICD10Result] = []
        for item in display:
            if isinstance(item, list) and len(item) >= 2:
                results.append(ICD10Result(code=item[0], description=item[1]))
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> Any:
        """HTTP GET with retry on transient errors."""
        try:
            resp = self._http.get(self._base, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("ICD-10 HTTP %s", exc.response.status_code)
            return []
        except Exception as exc:
            logger.warning("ICD-10 request failed: %s", exc)
            return []
