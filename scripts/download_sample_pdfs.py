"""Download sample policy PDFs to data/raw_pdfs/.

Replace the placeholder URLs below with real public policy document URLs
before running this script.
"""

import pathlib
import urllib.request

# ── Replace these with real public policy PDF URLs ────────────────────────────
SAMPLE_POLICIES: list[dict[str, str]] = [
    {
        "payer": "aetna",
        "name": "aetna_biologic_pa_policy.pdf",
        "url": "https://example.com/PLACEHOLDER_aetna_biologic.pdf",
    },
    {
        "payer": "cigna",
        "name": "cigna_oncology_drug_policy.pdf",
        "url": "https://example.com/PLACEHOLDER_cigna_oncology.pdf",
    },
    {
        "payer": "unitedhealthcare",
        "name": "uhc_specialty_drug_policy.pdf",
        "url": "https://example.com/PLACEHOLDER_uhc_specialty.pdf",
    },
    {
        "payer": "humana",
        "name": "humana_step_therapy_policy.pdf",
        "url": "https://example.com/PLACEHOLDER_humana_step.pdf",
    },
    {
        "payer": "anthem",
        "name": "anthem_quantity_limit_policy.pdf",
        "url": "https://example.com/PLACEHOLDER_anthem_ql.pdf",
    },
]

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw_pdfs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for policy in SAMPLE_POLICIES:
        dest = OUTPUT_DIR / policy["name"]
        if dest.exists():
            print(f"[skip] {dest.name} already downloaded.")
            continue
        print(f"[download] {policy['payer']} → {dest.name} ...")
        try:
            urllib.request.urlretrieve(policy["url"], dest)  # noqa: S310
            print(f"  ✓ saved to {dest}")
        except Exception as exc:
            print(f"  ✗ failed: {exc}")


if __name__ == "__main__":
    main()
