"""Create Neo4j constraints and indexes.

Usage
-----
python scripts/init_neo4j.py
"""

from __future__ import annotations

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


async def _run() -> None:
    # Late import so the script is runnable from the repo root without install
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    from dotenv import load_dotenv
    load_dotenv()

    from src.graph.client import Neo4jClient
    from src.graph.schema import init_schema

    async with Neo4jClient.from_settings() as client:
        await init_schema(client)
        logger.info("Done. Neo4j schema is ready.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
