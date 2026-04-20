"""PolicyLens — Streamlit demo UI.

Three tabs:
  Query   — natural-language questions with answer + citation trace
  Browse  — explore policies by payer, drug, or therapeutic area
  Ingest  — upload a new PDF and watch the pipeline run
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("POLICY_LENS_API", "http://localhost:8000")

st.set_page_config(
    page_title="PolicyLens",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("💊 PolicyLens")
    st.caption("Medical Benefit Drug Policy Tracker")
    st.divider()

    api_url = st.text_input("API URL", value=API_BASE)

    # Live health check
    try:
        r = httpx.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            st.success("API online", icon="✅")
        else:
            st.warning(f"API returned {r.status_code}", icon="⚠️")
    except Exception:
        st.error("API unreachable", icon="🔴")

    st.divider()
    st.markdown(
        "**Useful queries to try:**\n"
        "- *Does Aetna cover adalimumab for RA?*\n"
        "- *What are the PA criteria for Humira under Cigna PPO?*\n"
        "- *Compare GLP-1 coverage between Aetna and Humana*\n"
        "- *Which payers updated policies in 2024?*"
    )

# ── Tab layout ────────────────────────────────────────────────────────────────

tab_query, tab_browse, tab_ingest = st.tabs(["🔍 Query", "📋 Browse", "📤 Ingest"])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Query
# ════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.header("Ask a policy question")
    st.caption(
        "Claude will search the knowledge graph and return a cited answer, "
        "showing every tool call it made."
    )

    question = st.text_area(
        "Your question",
        placeholder="Does Aetna cover pembrolizumab for non-small cell lung cancer?",
        height=90,
    )
    max_rounds = st.slider("Max reasoning rounds", min_value=1, max_value=12, value=6)

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Reasoning over policies…"):
            try:
                resp = httpx.post(
                    f"{api_url}/query",
                    json={"question": question, "max_tool_rounds": max_rounds},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                st.error(f"API error {exc.response.status_code}: {exc.response.text}")
                data = None
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                data = None

        if data:
            st.subheader("Answer")
            st.markdown(data["answer"])

            if data.get("tool_calls"):
                st.divider()
                st.subheader(f"Reasoning trace — {len(data['tool_calls'])} tool call(s)")
                for i, tc in enumerate(data["tool_calls"], 1):
                    with st.expander(f"Step {i}: `{tc['tool_name']}`"):
                        col_in, col_out = st.columns(2)
                        with col_in:
                            st.markdown("**Input**")
                            st.json(tc["tool_input"])
                        with col_out:
                            st.markdown("**Result**")
                            result = tc["tool_result"]
                            if isinstance(result, str):
                                try:
                                    result = json.loads(result)
                                except Exception:
                                    pass
                            st.json(result)

            st.caption(f"Model: `{data.get('model', 'unknown')}`")


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Browse
# ════════════════════════════════════════════════════════════════════════════

with tab_browse:
    st.header("Browse ingested policies")

    browse_mode = st.radio(
        "Browse by",
        ["All policies", "Payer", "Drug name", "Therapeutic class"],
        horizontal=True,
    )

    # Build filter params
    params: dict[str, str] = {}
    drug_filter = ""
    class_filter = ""

    if browse_mode == "Payer":
        # First load payer list
        try:
            payer_resp = httpx.get(f"{api_url}/policies", timeout=10)
            all_pols = payer_resp.json() if payer_resp.status_code == 200 else []
            payer_names = sorted({p.get("payer", "") for p in all_pols if p.get("payer")})
        except Exception:
            payer_names = []
        selected_payer = st.selectbox("Select payer", [""] + payer_names)
        if selected_payer:
            params["payer_id"] = selected_payer.lower().replace(" ", "_")

    elif browse_mode == "Drug name":
        drug_filter = st.text_input("Drug name", placeholder="adalimumab")

    elif browse_mode == "Therapeutic class":
        class_filter = st.text_input("Drug class", placeholder="TNF inhibitor")

    if st.button("Search", key="browse_search"):
        with st.spinner("Searching…"):
            policies: list[dict[str, Any]] = []

            if browse_mode in ("All policies", "Payer"):
                try:
                    r = httpx.get(f"{api_url}/policies", params=params, timeout=15)
                    r.raise_for_status()
                    policies = r.json()
                except Exception as exc:
                    st.error(str(exc))

            elif drug_filter:
                try:
                    r = httpx.post(
                        f"{api_url}/query",
                        json={
                            "question": f"List all policies for drug '{drug_filter}' with policy_id, payer, and effective_date.",
                            "max_tool_rounds": 3,
                        },
                        timeout=60,
                    )
                    r.raise_for_status()
                    st.markdown(r.json().get("answer", ""))
                except Exception as exc:
                    st.error(str(exc))

            elif class_filter:
                try:
                    r = httpx.post(
                        f"{api_url}/query",
                        json={
                            "question": f"Find all policies covering drugs in class '{class_filter}'. List policy_id, payer, drugs.",
                            "max_tool_rounds": 3,
                        },
                        timeout=60,
                    )
                    r.raise_for_status()
                    st.markdown(r.json().get("answer", ""))
                except Exception as exc:
                    st.error(str(exc))

        if policies:
            st.success(f"{len(policies)} polic{'y' if len(policies) == 1 else 'ies'} found.")
            for pol in policies:
                with st.expander(
                    f"**{pol.get('title', pol.get('policy_id', '?'))}** — {pol.get('payer', '?')}  "
                    f"(effective {pol.get('effective_date', 'unknown')})"
                ):
                    st.markdown(f"**Policy ID:** `{pol.get('policy_id')}`")
                    if pol.get("version"):
                        st.markdown(f"**Version:** {pol['version']}")

                    # Load full detail on demand
                    if st.button("Load full detail", key=f"detail_{pol.get('policy_id')}"):
                        with st.spinner("Loading…"):
                            try:
                                dr = httpx.get(
                                    f"{api_url}/policies/{pol['policy_id']}", timeout=15
                                )
                                dr.raise_for_status()
                                detail = dr.json()
                                col_d, col_i, col_c = st.columns(3)
                                with col_d:
                                    st.markdown("**Drugs**")
                                    for d in (detail.get("drugs") or []):
                                        st.markdown(f"- {d.get('name', d)}")
                                with col_i:
                                    st.markdown("**Indications**")
                                    for ind in (detail.get("indications") or []):
                                        st.markdown(f"- {ind.get('name', ind)}")
                                with col_c:
                                    st.markdown("**Criteria**")
                                    for crit in (detail.get("criteria") or []):
                                        st.markdown(
                                            f"- **{crit.get('type', '?')}**: "
                                            f"{crit.get('description', '')[:80]}…"
                                        )
                            except Exception as exc:
                                st.error(str(exc))

        elif browse_mode in ("All policies", "Payer") and not policies:
            st.info("No policies found. Ingest some PDFs first.")


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Ingest
# ════════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.header("Ingest a new policy PDF")
    st.caption(
        "Upload a medical benefit drug policy PDF. "
        "The pipeline will extract structured data using Claude vision, "
        "populate the graph, and index excerpts for semantic search."
    )

    uploaded = st.file_uploader("Policy PDF", type=["pdf"])
    supersedes_id = st.text_input(
        "Supersedes policy_id (optional)",
        placeholder="aetna_biologic_pa_v1_abc12345",
        help="If this document replaces an existing policy, enter the old policy_id here.",
    )

    if uploaded and st.button("Ingest", type="primary"):
        progress = st.progress(0, text="Uploading…")

        with st.spinner("Running extraction pipeline…"):
            try:
                progress.progress(10, text="Sending to API…")
                resp = httpx.post(
                    f"{api_url}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    data={"supersedes": supersedes_id},
                    timeout=300,
                )
                progress.progress(90, text="Processing response…")
                resp.raise_for_status()
                result = resp.json()
                progress.progress(100, text="Done!")

                st.success(f"Policy ingested successfully: `{result['policy_id']}`")

                col1, col2, col3 = st.columns(3)
                col1.metric("Drugs found",       result["drugs_found"])
                col2.metric("Indications found",  result["indications_found"])
                col3.metric("Criteria found",     result["criteria_found"])

                st.markdown(f"**Payer:** {result['payer']}")
                st.markdown(f"**Title:** {result['title']}")
                st.markdown(f"**Policy ID:** `{result['policy_id']}`")

                if result.get("warnings"):
                    with st.expander(f"⚠️ {len(result['warnings'])} extraction warning(s)"):
                        for w in result["warnings"]:
                            st.warning(w)

            except httpx.HTTPStatusError as exc:
                progress.empty()
                st.error(f"Ingestion failed ({exc.response.status_code}): {exc.response.text}")
            except Exception as exc:
                progress.empty()
                st.error(f"Ingestion failed: {exc}")
