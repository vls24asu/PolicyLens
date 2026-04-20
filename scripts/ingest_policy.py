"""CLI to ingest a single policy PDF end-to-end.

Usage
-----
python scripts/ingest_policy.py --pdf data/raw_pdfs/aetna_biologic_pa.pdf
python scripts/ingest_policy.py --pdf policy.pdf --supersedes old_policy_id_here
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


async def _run(pdf_path: Path, supersedes: str | None) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from dotenv import load_dotenv
    load_dotenv()

    import anthropic
    from src.config import settings
    from src.graph.client import Neo4jClient
    from src.ingestion.graph_builder import GraphBuilder
    from src.ingestion.vlm_extractor import VLMExtractor

    logger.info("Ingesting: %s", pdf_path)

    # 1. Extract structured data from the PDF
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    extractor = VLMExtractor(client=client, model=settings.vlm_model)
    extracted = extractor.extract(pdf_path)

    logger.info(
        "Extracted: policy_id=%s, drugs=%d, indications=%d, criteria=%d",
        extracted.policy.policy_id,
        len(extracted.drugs),
        len(extracted.indications),
        len(extracted.criteria),
    )
    if extracted.extraction_warnings:
        for w in extracted.extraction_warnings:
            logger.warning("  Extraction warning: %s", w)

    # 2. Write to Neo4j
    async with Neo4jClient.from_settings() as neo4j:
        builder = GraphBuilder(neo4j)
        await builder.build(extracted)
        if supersedes:
            await builder.mark_supersedes(extracted.policy.policy_id, supersedes)

    logger.info("Done. policy_id=%s is now in the graph.", extracted.policy.policy_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a policy PDF into the graph.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument(
        "--supersedes",
        default=None,
        metavar="POLICY_ID",
        help="policy_id that this document supersedes (optional).",
    )
    args = parser.parse_args()
    asyncio.run(_run(Path(args.pdf), args.supersedes))


if __name__ == "__main__":
    main()
