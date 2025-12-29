"""
SDK MCP Server for Claude RAG Tools.

This server uses Claude Agent SDK to register tools in-process,
eliminating the need for external subprocess.
"""

from claude_agent_sdk import create_sdk_mcp_server

from .sdk_tools import (
    count_documents,
    # AgentFS Filesystem Tools (3)
    create_file,
    get_document,
    get_health,
    # Metrics Tools (2)
    get_metrics,
    get_state,
    ingest_document,
    list_files,
    list_sources,
    list_states,
    read_file,
    # RAG Search Tools (6)
    search_documents,
    search_hybrid,
    # AgentFS KV Store Tools (3)
    set_state,
)


def create_rag_sdk_server():
    """Create SDK MCP server with all RAG tools.

    Returns:
        SDK MCP server ready for use with ClaudeAgentOptions
    """
    return create_sdk_mcp_server(
        name="rag-tools",
        version="1.0.0",
        tools=[
            # RAG Search (6 tools)
            search_documents,
            search_hybrid,
            get_document,
            list_sources,
            count_documents,
            ingest_document,
            # Filesystem (3 tools)
            create_file,
            read_file,
            list_files,
            # KV Store (3 tools)
            set_state,
            get_state,
            list_states,
            # Metrics (2 tools)
            get_metrics,
            get_health,
        ],
    )


# Convenience: list of all allowed tool names
RAG_ALLOWED_TOOLS = [
    "mcp__rag-tools__search_documents",
    "mcp__rag-tools__search_hybrid",
    "mcp__rag-tools__get_document",
    "mcp__rag-tools__list_sources",
    "mcp__rag-tools__count_documents",
    "mcp__rag-tools__ingest_document",
    "mcp__rag-tools__create_file",
    "mcp__rag-tools__read_file",
    "mcp__rag-tools__list_files",
    "mcp__rag-tools__set_state",
    "mcp__rag-tools__get_state",
    "mcp__rag-tools__list_states",
    "mcp__rag-tools__get_metrics",
    "mcp__rag-tools__get_health",
]
