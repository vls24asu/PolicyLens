"""End-to-end smoke test for PolicyLens.

Requires a running API server (default http://localhost:8000) and a real PDF.
Usage:
    python scripts/smoke_test.py [--api-url URL] [--pdf PATH] [--question TEXT]

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import httpx

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[34mINFO\033[0m"

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f": {detail}" if detail else ""))
        _failures.append(label)


def section(title: str) -> None:
    print(f"\n-- {title} " + "-" * max(0, 60 - len(title)))


# ── Stage checks ──────────────────────────────────────────────────────────────

def check_health(client: httpx.Client) -> bool:
    section("Health")
    try:
        r = client.get("/health", timeout=5)
        check("GET /health → 200", r.status_code == 200)
        check("/health body == {status: ok}", r.json() == {"status": "ok"})
        return True
    except Exception as exc:
        check("GET /health reachable", False, str(exc))
        return False


def check_list_policies_empty_ok(client: httpx.Client) -> None:
    section("List policies (may be empty before ingest)")
    try:
        r = client.get("/policies", timeout=10)
        check("GET /policies → 200", r.status_code == 200)
        check("GET /policies returns list", isinstance(r.json(), list))
    except Exception as exc:
        check("GET /policies", False, str(exc))


def check_ingest(client: httpx.Client, pdf_path: Path) -> str | None:
    section(f"Ingest — {pdf_path.name}")
    try:
        with pdf_path.open("rb") as f:
            r = client.post(
                "/ingest",
                files={"file": (pdf_path.name, f, "application/pdf")},
                timeout=300,
            )
        check("POST /ingest → 200", r.status_code == 200, r.text[:200])
        if r.status_code != 200:
            return None

        body = r.json()
        check("policy_id present", bool(body.get("policy_id")))
        check("title present", bool(body.get("title")))
        check("payer present", bool(body.get("payer")))
        check("drugs_found >= 0", isinstance(body.get("drugs_found"), int))
        check("criteria_found >= 0", isinstance(body.get("criteria_found"), int))
        print(f"  {INFO}  policy_id = {body.get('policy_id')}")
        print(f"  {INFO}  drugs={body.get('drugs_found')}  "
              f"indications={body.get('indications_found')}  "
              f"criteria={body.get('criteria_found')}")
        if body.get("warnings"):
            print(f"  {INFO}  {len(body['warnings'])} extraction warning(s)")
        return body.get("policy_id")
    except Exception as exc:
        check("POST /ingest", False, str(exc))
        return None


def check_list_policies_after_ingest(client: httpx.Client, policy_id: str) -> None:
    section("List policies after ingest")
    try:
        r = client.get("/policies", timeout=10)
        check("GET /policies → 200", r.status_code == 200)
        ids = [p.get("policy_id") for p in r.json()]
        check(f"policy_id '{policy_id}' in list", policy_id in ids)
    except Exception as exc:
        check("GET /policies after ingest", False, str(exc))


def check_get_policy(client: httpx.Client, policy_id: str) -> None:
    section("Get policy detail")
    try:
        r = client.get(f"/policies/{policy_id}", timeout=10)
        check(f"GET /policies/{policy_id} → 200", r.status_code == 200)
        body = r.json()
        check("detail has policy_id", body.get("policy_id") == policy_id)
    except Exception as exc:
        check("GET /policies/{id}", False, str(exc))

    # 404 for unknown
    try:
        r404 = client.get("/policies/__nonexistent__", timeout=10)
        check("GET /policies/__nonexistent__ → 404", r404.status_code == 404)
    except Exception as exc:
        check("404 for unknown policy", False, str(exc))


def check_query(client: httpx.Client, question: str) -> None:
    section("Query endpoint")
    try:
        r = client.post(
            "/query",
            json={"question": question, "max_tool_rounds": 6},
            timeout=120,
        )
        check("POST /query → 200", r.status_code == 200, r.text[:200])
        if r.status_code != 200:
            return

        body = r.json()
        check("answer is non-empty string", isinstance(body.get("answer"), str) and len(body["answer"]) > 0)
        check("tool_calls is list", isinstance(body.get("tool_calls"), list))
        check("model field present", bool(body.get("model")))

        n = len(body["tool_calls"])
        print(f"  {INFO}  {n} tool call(s) made")
        print(f"  {INFO}  model = {body.get('model')}")
        print(f"  {INFO}  answer (first 200 chars):")
        print(textwrap.indent(body["answer"][:200], "       "))
    except Exception as exc:
        check("POST /query", False, str(exc))


def check_ingest_rejection(client: httpx.Client) -> None:
    section("Non-PDF rejection")
    try:
        r = client.post(
            "/ingest",
            files={"file": ("doc.txt", b"plain text", "text/plain")},
            timeout=10,
        )
        check("non-PDF upload → 400", r.status_code == 400)
    except Exception as exc:
        check("non-PDF rejection", False, str(exc))


# ── Synthetic PDF fallback ─────────────────────────────────────────────────────

def _make_synthetic_pdf() -> Path:
    """Create a minimal valid PDF in a temp location."""
    try:
        import fitz
        import tempfile

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Medical Benefit Drug Policy\n"
            "Payer: SmokeTest Insurance\n"
            "Drug: Adalimumab (Humira)\n"
            "Indication: Rheumatoid Arthritis\n"
            "Prior authorization required.\n"
            "Effective Date: 2024-01-01",
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        return Path(tmp.name)
    except ImportError:
        print(f"  {INFO}  PyMuPDF not available — skipping ingest checks")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="PolicyLens end-to-end smoke test")
    parser.add_argument("--api-url", default="http://localhost:8000", metavar="URL")
    parser.add_argument("--pdf", default=None, metavar="PATH",
                        help="Path to a real policy PDF (optional; synthetic used if omitted)")
    parser.add_argument("--question", default="What drugs are covered by the ingested policy?",
                        metavar="TEXT")
    args = parser.parse_args()

    print(f"\nPolicyLens smoke test -> {args.api_url}\n")

    with httpx.Client(base_url=args.api_url) as client:
        # 1. Health
        alive = check_health(client)
        if not alive:
            print("\nAPI is unreachable — aborting smoke test.")
            return 1

        # 2. Initial policy list
        check_list_policies_empty_ok(client)

        # 3. Non-PDF rejection (doesn't need real API key)
        check_ingest_rejection(client)

        # 4. Ingest
        pdf_path: Path | None = Path(args.pdf) if args.pdf else _make_synthetic_pdf()
        policy_id: str | None = None

        if pdf_path and pdf_path.exists():
            policy_id = check_ingest(client, pdf_path)
        else:
            section("Ingest")
            print(f"  {INFO}  No PDF available — skipping ingest and downstream checks")

        # 5. Post-ingest checks
        if policy_id:
            check_list_policies_after_ingest(client, policy_id)
            check_get_policy(client, policy_id)
            check_query(client, args.question)
        else:
            section("Query (no ingest)")
            print(f"  {INFO}  Running query without prior ingest")
            check_query(client, args.question)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 62)
    if _failures:
        print(f"\n{FAIL}  {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"     • {f}")
        print()
        return 1
    else:
        print(f"\n{PASS}  All checks passed.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
