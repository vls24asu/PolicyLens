"""FastAPI backend for PolicyLens.

Endpoints
---------
POST /ingest          Upload a PDF and run the full extraction + graph pipeline.
POST /query           Natural-language question answered by Claude via MCP tools.
GET  /policies        List all ingested policies (optional ?payer_id= filter).
GET  /policies/{id}   Full detail for one policy.
GET  /health          Liveness check.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)

# ── Lazy singletons ───────────────────────────────────────────────────────────

_neo4j_client: Any = None
_excerpt_store: Any = None
_anthropic_client: Any = None


async def _get_neo4j() -> Any:
    global _neo4j_client
    if _neo4j_client is None:
        from src.graph.client import Neo4jClient
        _neo4j_client = Neo4jClient.from_settings()
        await _neo4j_client.connect()
    return _neo4j_client


def _get_store() -> Any:
    global _excerpt_store
    if _excerpt_store is None:
        from src.vector_store.excerpt_store import ExcerptStore
        _excerpt_store = ExcerptStore.from_settings()
    return _excerpt_store


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: ARG001
    yield
    if _neo4j_client is not None:
        await _neo4j_client.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PolicyLens API",
    description="Medical Benefit Drug Policy Tracker",
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    max_tool_rounds: int = 8


class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any


class QueryResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCall]
    model: str


class IngestResponse(BaseModel):
    policy_id: str
    title: str
    payer: str
    drugs_found: int
    indications_found: int
    criteria_found: int
    warnings: list[str]


# ── Tool registry ─────────────────────────────────────────────────────────────

def _build_tool_registry() -> dict[str, Any]:
    """Map tool names → callable functions from all MCP server modules."""
    from src.mcp_servers import (
        policy_graph_server,
        document_retrieval_server,
        reference_data_server,
        change_detection_server,
    )
    registry: dict[str, Any] = {}
    for mod in (
        policy_graph_server,
        document_retrieval_server,
        reference_data_server,
        change_detection_server,
    ):
        for name, tool in mod.mcp._tool_manager._tools.items():
            registry[name] = tool.fn
    return registry


def _build_anthropic_tools(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert FastMCP tool registry into Anthropic tool definitions."""
    from src.mcp_servers import (
        policy_graph_server,
        document_retrieval_server,
        reference_data_server,
        change_detection_server,
    )
    tools: list[dict[str, Any]] = []
    for mod in (
        policy_graph_server,
        document_retrieval_server,
        reference_data_server,
        change_detection_server,
    ):
        for name, tool in mod.mcp._tool_manager._tools.items():
            schema = tool.parameters or {"type": "object", "properties": {}}
            # Strip the pydantic/jsonschema $defs wrapper that FastMCP sometimes adds
            if "properties" not in schema:
                schema = {"type": "object", "properties": {}}
            tools.append({
                "name":         name,
                "description":  tool.description or name,
                "input_schema": schema,
            })
    return tools


async def _execute_tool(
    name: str,
    tool_input: dict[str, Any],
    registry: dict[str, Any],
) -> Any:
    """Call a tool function by name, awaiting it if async."""
    fn = registry.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        result = fn(**tool_input)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:
        logger.exception("Tool '%s' raised: %s", name, exc)
        return {"error": str(exc)}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    supersedes: str = Form(default=""),
) -> IngestResponse:
    """Upload a medical policy PDF and ingest it into the graph.

    Runs VLM extraction, graph population, and excerpt indexing in sequence.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    import anthropic as _anthropic
    from src.ingestion.vlm_extractor import VLMExtractor
    from src.ingestion.graph_builder import GraphBuilder

    # Save upload to a temp file so PyMuPDF can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        # 1. VLM extraction
        client = _get_anthropic()
        extractor = VLMExtractor(client=client, model=settings.vlm_model)
        extracted = extractor.extract(tmp_path)

        # 2. Graph population
        neo4j = await _get_neo4j()
        builder = GraphBuilder(neo4j)
        await builder.build(extracted)
        if supersedes.strip():
            await builder.mark_supersedes(extracted.policy.policy_id, supersedes.strip())

        # 3. Index excerpts
        store = _get_store()
        store.add_from_policy(extracted)

        return IngestResponse(
            policy_id=extracted.policy.policy_id,
            title=extracted.policy.title,
            payer=extracted.payer.name,
            drugs_found=len(extracted.drugs),
            indications_found=len(extracted.indications),
            criteria_found=len(extracted.criteria),
            warnings=extracted.extraction_warnings,
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question using Claude with MCP tools.

    Claude decides which tools to call, the backend executes them, and the
    final answer with a full tool-call trace is returned.
    """
    registry = _build_tool_registry()
    # Inject lazy clients into MCP server modules so tools can use them
    await _prime_mcp_clients()

    tools = _build_anthropic_tools(registry)
    client = _get_anthropic()

    messages: list[dict[str, Any]] = [{"role": "user", "content": request.question}]
    tool_calls_log: list[ToolCall] = []

    system = (
        "You are a medical benefit drug policy analyst. "
        "Use the available tools to research the question, then provide a clear, "
        "cited answer. Always mention the payer, plan name, and policy effective date "
        "when reporting coverage information."
    )

    for _ in range(request.max_tool_rounds):
        response = client.messages.create(
            model=settings.vlm_model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Collect text and tool use blocks
        assistant_content: list[dict[str, Any]] = []
        tool_use_blocks: list[Any] = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
                assistant_content.append({
                    "type":  "tool_use",
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            # Extract the final text answer
            final_text = " ".join(
                b["text"] for b in assistant_content if b.get("type") == "text"
            ).strip()
            return QueryResponse(
                answer=final_text or "(no text response)",
                tool_calls=tool_calls_log,
                model=response.model,
            )

        # Execute all tool calls and append results
        tool_results: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            result = await _execute_tool(block.name, dict(block.input), registry)
            tool_calls_log.append(ToolCall(
                tool_name=block.name,
                tool_input=dict(block.input),
                tool_result=result,
            ))
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     _serialise(result),
            })

        messages.append({"role": "user", "content": tool_results})

    raise HTTPException(
        status_code=500,
        detail=f"Claude did not produce a final answer within {request.max_tool_rounds} tool rounds.",
    )


@app.get("/policies")
async def list_policies(payer_id: str = "") -> list[dict[str, Any]]:
    """List all ingested policies, optionally filtered by payer_id."""
    from src.graph import queries
    neo4j = await _get_neo4j()
    return await queries.list_policies(neo4j, payer_id=payer_id.strip() or None)


@app.get("/policies/{policy_id}")
async def get_policy(policy_id: str) -> dict[str, Any]:
    """Return full detail for a single policy."""
    from src.graph import queries
    neo4j = await _get_neo4j()
    detail = await queries.get_policy_details(neo4j, policy_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found.")
    return detail


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _prime_mcp_clients() -> None:
    """Pre-connect the lazy Neo4j clients inside each MCP server module."""
    import src.mcp_servers.policy_graph_server       as pgs
    import src.mcp_servers.document_retrieval_server as drs
    import src.mcp_servers.change_detection_server   as cds

    neo4j = await _get_neo4j()
    store = _get_store()

    pgs._neo4j_client  = neo4j
    drs._neo4j_client  = neo4j
    drs._excerpt_store = store
    cds._neo4j_client  = neo4j


def _serialise(value: Any) -> str:
    """Convert a tool result to a JSON string for the Anthropic messages API."""
    import json
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)
