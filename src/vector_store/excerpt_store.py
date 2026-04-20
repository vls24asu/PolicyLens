"""ChromaDB wrapper for policy source excerpts.

Stores verbatim text excerpts from policy PDFs and supports semantic search
with page-level citation metadata.  Uses ChromaDB's built-in embedding
function (all-MiniLM-L6-v2 via sentence-transformers) by default so the store
works offline without an additional API key.

Typical usage
-------------
store = ExcerptStore.from_settings()
store.add_from_policy(extracted_policy)

results = store.search("prior authorization requirements", policy_id="pol_001")
for r in results:
    print(r.text, f"(page {r.page_number})")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "policy_excerpts"


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class ExcerptResult:
    """A single search result from the excerpt store."""

    excerpt_id: str
    policy_id: str
    text: str
    page_number: int
    topic: str | None
    distance: float                        # lower = more similar
    relevance: float                       # 1 - distance, clamped to [0, 1]
    bbox: list[float] | None = field(default=None)

    @property
    def citation(self) -> str:
        """Human-readable citation string."""
        return f"Policy {self.policy_id}, page {self.page_number}"


# ── ExcerptStore ──────────────────────────────────────────────────────────────

class ExcerptStore:
    """Semantic search over policy source excerpts backed by ChromaDB.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB stores its data.  Pass ``":memory:"`` in
        tests for an in-process ephemeral store.
    collection_name:
        ChromaDB collection name.  Override in tests to avoid cross-test
        contamination.
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = self._make_client(persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ExcerptStore ready: collection=%s, items=%d",
            collection_name,
            self._collection.count(),
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, collection_name: str = _COLLECTION_NAME) -> "ExcerptStore":
        """Construct from the global Settings object."""
        from src.config import settings
        return cls(persist_dir=settings.chroma_persist_dir, collection_name=collection_name)

    # ── Write operations ──────────────────────────────────────────────────────

    def add_from_policy(self, extracted: Any) -> int:
        """Add all SourceExcerpts from an ExtractedPolicy.

        Parameters
        ----------
        extracted:
            An ``ExtractedPolicy`` instance.

        Returns
        -------
        int
            Number of excerpts added (skips duplicates by excerpt_id).
        """
        from src.models.policy import SourceExcerpt
        excerpts: list[SourceExcerpt] = extracted.excerpts
        if not excerpts:
            logger.debug("add_from_policy: no excerpts in policy %s", extracted.policy.policy_id)
            return 0
        return self.add_excerpts(excerpts)

    def add_excerpts(self, excerpts: list[Any]) -> int:
        """Upsert a list of SourceExcerpt objects into the collection.

        Returns the number of documents written.
        """
        if not excerpts:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for exc in excerpts:
            if not exc.text.strip():
                continue
            ids.append(exc.excerpt_id)
            documents.append(exc.text)
            meta: dict[str, Any] = {
                "policy_id":   exc.policy_id,
                "page_number": exc.page_number,
            }
            if exc.topic:
                meta["topic"] = exc.topic
            if exc.bbox:
                # ChromaDB metadata values must be str/int/float/bool —
                # serialise the list as a comma-separated string.
                meta["bbox"] = ",".join(str(v) for v in exc.bbox)
            metadatas.append(meta)

        if not ids:
            return 0

        # upsert is idempotent — safe to call multiple times for the same IDs
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.debug("ExcerptStore: upserted %d excerpts", len(ids))
        return len(ids)

    def delete_policy(self, policy_id: str) -> int:
        """Remove all excerpts belonging to a policy.

        Returns the number of documents deleted.
        """
        results = self._collection.get(where={"policy_id": policy_id})
        ids_to_delete: list[str] = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("ExcerptStore: deleted %d excerpts for policy %s", len(ids_to_delete), policy_id)
        return len(ids_to_delete)

    # ── Read operations ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        policy_id: str | None = None,
        n_results: int = 5,
        topic: str | None = None,
    ) -> list[ExcerptResult]:
        """Semantic search over stored excerpts.

        Parameters
        ----------
        query:
            Natural-language query string.
        policy_id:
            If provided, restrict results to this policy.
        n_results:
            Maximum number of results to return.
        topic:
            If provided, restrict results to excerpts with this topic label.

        Returns
        -------
        list[ExcerptResult]
            Ranked by cosine similarity (closest first).
        """
        if self._collection.count() == 0:
            return []

        where: dict[str, Any] | None = self._build_where(policy_id, topic)

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        return self._parse_results(raw)

    def get_by_policy(
        self,
        policy_id: str,
        topic: str | None = None,
        limit: int = 100,
    ) -> list[ExcerptResult]:
        """Return all stored excerpts for a policy, ordered by page number.

        Parameters
        ----------
        policy_id:
            The policy to retrieve excerpts for.
        topic:
            Optional topic filter.
        limit:
            Maximum number of results.
        """
        where = self._build_where(policy_id, topic)
        raw = self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        results: list[ExcerptResult] = []
        ids = raw.get("ids", [])
        docs = raw.get("documents", [])
        metas = raw.get("metadatas", [])
        for excerpt_id, text, meta in zip(ids, docs, metas):
            results.append(self._make_result(excerpt_id, text, meta, distance=0.0))
        results.sort(key=lambda r: r.page_number)
        return results

    def count(self, policy_id: str | None = None) -> int:
        """Return number of stored excerpts, optionally filtered by policy."""
        if policy_id is None:
            return self._collection.count()
        raw = self._collection.get(where={"policy_id": policy_id})
        return len(raw.get("ids", []))

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_client(persist_dir: str) -> chromadb.ClientAPI:
        if persist_dir == ":memory:":
            return chromadb.EphemeralClient()
        return chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    @staticmethod
    def _build_where(
        policy_id: str | None,
        topic: str | None,
    ) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []
        if policy_id:
            conditions.append({"policy_id": {"$eq": policy_id}})
        if topic:
            conditions.append({"topic": {"$eq": topic}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _make_result(
        excerpt_id: str,
        text: str,
        meta: dict[str, Any],
        distance: float,
    ) -> ExcerptResult:
        bbox_raw = meta.get("bbox")
        bbox = [float(v) for v in bbox_raw.split(",")] if bbox_raw else None
        relevance = max(0.0, min(1.0, 1.0 - distance))
        return ExcerptResult(
            excerpt_id=excerpt_id,
            policy_id=meta.get("policy_id", ""),
            text=text,
            page_number=int(meta.get("page_number", 0)),
            topic=meta.get("topic"),
            distance=distance,
            relevance=relevance,
            bbox=bbox,
        )

    def _parse_results(self, raw: dict[str, Any]) -> list[ExcerptResult]:
        results: list[ExcerptResult] = []
        ids_batch    = raw.get("ids",       [[]])[0]
        docs_batch   = raw.get("documents", [[]])[0]
        metas_batch  = raw.get("metadatas", [[]])[0]
        dists_batch  = raw.get("distances", [[]])[0]
        for excerpt_id, text, meta, dist in zip(ids_batch, docs_batch, metas_batch, dists_batch):
            results.append(self._make_result(excerpt_id, text, meta, dist))
        return results
