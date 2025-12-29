"""MCP Server for Claude RAG SDK.

Two server implementations:
- sdk_server.py: In-process SDK MCP server (recommended)
- server.py: External FastMCP server (legacy)
"""

# Export SDK server (in-process, recommended)
from .sdk_server import RAG_ALLOWED_TOOLS, create_rag_sdk_server

# Export tools for direct use
from .sdk_tools import (
    count_documents,
    create_file,
    get_document,
    get_health,
    get_metrics,
    get_state,
    ingest_document,
    list_files,
    list_sources,
    list_states,
    read_file,
    search_documents,
    search_hybrid,
    set_state,
)

__all__ = [
    # Server factory
    "create_rag_sdk_server",
    "RAG_ALLOWED_TOOLS",
    # Tools
    "search_documents",
    "search_hybrid",
    "get_document",
    "list_sources",
    "count_documents",
    "ingest_document",
    "create_file",
    "read_file",
    "list_files",
    "set_state",
    "get_state",
    "list_states",
    "get_metrics",
    "get_health",
]
