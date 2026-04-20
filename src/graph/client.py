"""Async Neo4j client wrapper."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession, Record

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Thin async wrapper around the official Neo4j driver.

    Usage
    -----
    async with Neo4jClient.from_settings() as client:
        records = await client.run("RETURN 1 AS n")
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: AsyncDriver | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the driver and verify connectivity."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        await self._driver.verify_connectivity()
        logger.info("Neo4j connected: %s", self._uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    async def __aenter__(self) -> "Neo4jClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    @classmethod
    def from_settings(cls) -> "Neo4jClient":
        """Construct from the global Settings object."""
        from src.config import settings
        return cls(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)

    # ── Session helper ────────────────────────────────────────────────────────

    @asynccontextmanager
    async def session(self, database: str = "neo4j") -> AsyncGenerator[AsyncSession, None]:
        if self._driver is None:
            raise RuntimeError("Neo4jClient is not connected. Call connect() first.")
        async with self._driver.session(database=database) as s:
            yield s

    # ── Query helpers ─────────────────────────────────────────────────────────

    async def run(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[Record]:
        """Execute a Cypher statement and return all records."""
        async with self.session(database=database) as s:
            result = await s.run(cypher, parameters or {})
            return await result.data()

    async def run_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[Record]:
        """Execute a write transaction."""
        async def _tx(tx: Any) -> list[Record]:
            result = await tx.run(cypher, parameters or {})
            return await result.data()

        async with self.session(database=database) as s:
            return await s.execute_write(_tx)

    async def run_write_many(
        self,
        statements: list[tuple[str, dict[str, Any]]],
        database: str = "neo4j",
    ) -> None:
        """Execute multiple write statements in a single transaction."""
        async def _tx(tx: Any) -> None:
            for cypher, params in statements:
                await tx.run(cypher, params)

        async with self.session(database=database) as s:
            await s.execute_write(_tx)

    async def merge_node(
        self,
        label: str,
        merge_key: dict[str, Any],
        properties: dict[str, Any],
        database: str = "neo4j",
    ) -> None:
        """MERGE a node by key, then SET all properties."""
        key_clause = " AND ".join(f"n.{k} = $mk_{k}" for k in merge_key)
        set_clause = ", ".join(f"n.{k} = $p_{k}" for k in properties)
        prefixed_keys = {f"mk_{k}": v for k, v in merge_key.items()}
        prefixed_props = {f"p_{k}": v for k, v in properties.items()}
        cypher = (
            f"MERGE (n:{label} {{{', '.join(f'{k}: $mk_{k}' for k in merge_key)}}}) "
            f"SET {set_clause}"
        ) if set_clause else (
            f"MERGE (n:{label} {{{', '.join(f'{k}: $mk_{k}' for k in merge_key)}}})"
        )
        await self.run_write(cypher, {**prefixed_keys, **prefixed_props}, database)

    async def merge_relationship(
        self,
        from_label: str,
        from_key: dict[str, Any],
        to_label: str,
        to_key: dict[str, Any],
        rel_type: str,
        properties: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> None:
        """MERGE a relationship between two nodes identified by their keys."""
        from_match = ", ".join(f"{k}: $from_{k}" for k in from_key)
        to_match = ", ".join(f"{k}: $to_{k}" for k in to_key)
        params: dict[str, Any] = {
            **{f"from_{k}": v for k, v in from_key.items()},
            **{f"to_{k}": v for k, v in to_key.items()},
        }
        if properties:
            prop_set = " SET r." + ", r.".join(f"{k} = $rp_{k}" for k in properties)
            params.update({f"rp_{k}": v for k, v in properties.items()})
        else:
            prop_set = ""

        cypher = (
            f"MATCH (a:{from_label} {{{from_match}}}) "
            f"MATCH (b:{to_label} {{{to_match}}}) "
            f"MERGE (a)-[r:{rel_type}]->(b)"
            f"{prop_set}"
        )
        await self.run_write(cypher, params, database)
