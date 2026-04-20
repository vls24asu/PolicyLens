"""Launch any PolicyLens MCP server over stdio.

Usage
-----
python scripts/run_mcp_server.py policy_graph
python scripts/run_mcp_server.py document_retrieval
python scripts/run_mcp_server.py reference_data
python scripts/run_mcp_server.py change_detection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

SERVERS = {
    "policy_graph":        "src.mcp_servers.policy_graph_server",
    "document_retrieval":  "src.mcp_servers.document_retrieval_server",
    "reference_data":      "src.mcp_servers.reference_data_server",
    "change_detection":    "src.mcp_servers.change_detection_server",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a PolicyLens MCP server over stdio.")
    parser.add_argument("server", choices=list(SERVERS), help="Server to run.")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    import importlib
    module = importlib.import_module(SERVERS[args.server])
    module.mcp.run()


if __name__ == "__main__":
    main()
