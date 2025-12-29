"""
SDK MCP Tools for Claude RAG.

Custom tools using Claude Agent SDK @tool decorator.
These tools use AgentFS as backend and follow SDK patterns.
"""

import json
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

# =============================================================================
# GLOBAL STATE AND HELPERS
# =============================================================================

# Global RAG instance (initialized lazily)
_rag = None
_session_id = None


def _get_session_id() -> str:
    """Get current session ID."""
    global _session_id

    if _session_id:
        return _session_id

    # Try to read from file
    session_file = Path.home() / ".claude" / ".agentfs" / "current_session"
    if session_file.exists():
        try:
            return session_file.read_text().strip()
        except OSError:
            pass  # File read failed, use default

    return "default"


async def _get_rag():
    """Get or create RAG instance."""
    global _rag, _session_id

    if _rag is None:
        from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions

        _session_id = _get_session_id()

        options = ClaudeRAGOptions(
            id=_session_id,
            enable_reranking=True,
            enable_prompt_guard=True,
        )

        _rag = await ClaudeRAG.open(options)
        print(f"[SDK Tools] RAG initialized: {_session_id}")

    return _rag


# =============================================================================
# RAG SEARCH TOOLS
# =============================================================================


@tool("search_documents", "Semantic search in indexed documents", {
    "query": str,
    "top_k": int,
    "use_reranking": bool,
})
async def search_documents(args: dict[str, Any]) -> dict[str, Any]:
    """
    Semantic search in indexed documents.

    Args:
        query: Search query
        top_k: Number of results (default 5)
        use_reranking: Apply reranking for better precision (default True)

    Returns:
        List of relevant documents with source, content and score
    """
    query = args["query"]
    top_k = args.get("top_k", 5)
    use_reranking = args.get("use_reranking", True)
    start_time = time.perf_counter()

    try:
        rag = await _get_rag()
        results = await rag.search(query, top_k=top_k, use_reranking=use_reranking)

        response = [
            {
                "doc_id": r.doc_id,
                "source": r.source,
                "type": r.doc_type,
                "content": r.content[:1000] if r.content else "",
                "similarity": r.similarity,
                "rerank_score": r.rerank_score,
                "rank": r.rank,
            }
            for r in results
        ]

        # Track tool call
        await rag.tools.record(
            "search_documents",
            started_at=start_time,
            completed_at=time.perf_counter(),
            parameters={"query": query[:100], "top_k": top_k},
            result={"count": len(response)},
        )

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("search_hybrid", "Hybrid search combining BM25 and vector search", {
    "query": str,
    "top_k": int,
    "vector_weight": float,
})
async def search_hybrid(args: dict[str, Any]) -> dict[str, Any]:
    """
    Hybrid search combining BM25 (lexical) and vector search.

    Args:
        query: Search query
        top_k: Number of results (default 5)
        vector_weight: Weight for vector search (0-1, default 0.7)

    Returns:
        List of documents with hybrid scores
    """
    query = args["query"]
    top_k = args.get("top_k", 5)
    vector_weight = args.get("vector_weight", 0.7)

    try:
        rag = await _get_rag()
        results = await rag.search_hybrid(
            query,
            top_k=top_k,
            vector_weight=vector_weight,
        )

        response = [
            {
                "doc_id": r.doc_id,
                "source": r.source,
                "type": r.doc_type,
                "content": r.content[:1000] if r.content else "",
                "vector_score": r.vector_score,
                "bm25_score": r.bm25_score,
                "hybrid_score": r.hybrid_score,
                "rank": r.rank,
            }
            for r in results
        ]

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("get_document", "Get full document by ID", {"doc_id": int})
async def get_document(args: dict[str, Any]) -> dict[str, Any]:
    """
    Get full document by ID.

    Args:
        doc_id: Document ID

    Returns:
        Complete document with all fields
    """
    doc_id = args["doc_id"]

    try:
        rag = await _get_rag()
        doc = await rag.get_document(doc_id)

        if doc is None:
            return {
                "content": [{"type": "text", "text": f"Documento {doc_id} não encontrado"}],
                "is_error": True,
            }

        return {
            "content": [{"type": "text", "text": json.dumps(doc, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("list_sources", "List all available document sources", {})
async def list_sources(args: dict[str, Any]) -> dict[str, Any]:
    """
    List all available document sources.

    Returns:
        List of documents with name and type
    """
    try:
        rag = await _get_rag()
        sources = await rag.list_sources()

        return {
            "content": [{"type": "text", "text": json.dumps(sources, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("count_documents", "Count documents and embeddings", {})
async def count_documents(args: dict[str, Any]) -> dict[str, Any]:
    """
    Count documents and embeddings.

    Returns:
        Database statistics
    """
    try:
        rag = await _get_rag()
        stats = await rag.stats()
        doc_stats = stats.get("documents", {})

        return {
            "content": [{"type": "text", "text": json.dumps(doc_stats, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("ingest_document", "Add a document to the knowledge base", {
    "content": str,
    "source": str,
    "metadata": dict,
})
async def ingest_document(args: dict[str, Any]) -> dict[str, Any]:
    """
    Add a document to the knowledge base.

    Args:
        content: Document text content
        source: Source identifier (e.g., "manual_entry", "api", file name)
        metadata: Optional metadata dict (tags, category, etc.)

    Returns:
        Info about ingested document including doc_id
    """
    content = args["content"]
    source = args["source"]
    metadata = args.get("metadata")
    start_time = time.perf_counter()

    try:
        rag = await _get_rag()
        result = await rag.add_text(content, source, metadata=metadata)

        # Track tool call
        await rag.tools.record(
            "ingest_document",
            started_at=start_time,
            completed_at=time.perf_counter(),
            parameters={"source": source, "content_length": len(content)},
            result={"doc_id": result.doc_id if hasattr(result, "doc_id") else None},
        )

        response = {
            "success": True,
            "doc_id": result.doc_id if hasattr(result, "doc_id") else None,
            "source": source,
            "content_length": len(content),
            "message": f"Documento '{source}' ingerido com sucesso",
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


# =============================================================================
# AGENTFS FILESYSTEM TOOLS
# =============================================================================


@tool("create_file", "Create file in agent filesystem", {
    "path": str,
    "content": str,
})
async def create_file(args: dict[str, Any]) -> dict[str, Any]:
    """
    Create file in agent filesystem.

    Args:
        path: File name (e.g., "report.txt")
        content: File content

    Returns:
        Info about created file
    """
    path = args["path"]
    content = args["content"]

    try:
        rag = await _get_rag()
        await rag.fs.write_file(f"/{path}", content)

        # Also save to physical filesystem organized by session
        session_id = _get_session_id()
        if session_id and session_id != "default":
            # Get backend directory
            backend_dir = Path(__file__).parent.parent.parent
            artifacts_dir = backend_dir / "artifacts" / session_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # Extract filename from path (remove leading / if present)
            filename = path.lstrip("/")
            physical_file = artifacts_dir / filename
            physical_file.write_text(content, encoding="utf-8")

        response = {
            "success": True,
            "path": path,
            "size": len(content),
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("read_file", "Read file from agent filesystem", {"path": str})
async def read_file(args: dict[str, Any]) -> dict[str, Any]:
    """
    Read file from agent filesystem.

    Args:
        path: File name

    Returns:
        File content
    """
    path = args["path"]

    try:
        rag = await _get_rag()
        content = await rag.fs.read_file(f"/{path}")

        response = {
            "success": True,
            "path": path,
            "content": content,
            "size": len(content) if content else 0,
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("list_files", "List files in agent filesystem", {"directory": str})
async def list_files(args: dict[str, Any]) -> dict[str, Any]:
    """
    List files in agent filesystem.

    Args:
        directory: Directory to list (default: root)

    Returns:
        List of files and directories
    """
    directory = args.get("directory", "/")

    try:
        rag = await _get_rag()
        entries = await rag.fs.readdir(directory)

        files = []
        for entry in entries:
            files.append({
                "name": entry.name,
                "size": entry.size if hasattr(entry, "size") else 0,
                "is_dir": entry.is_dir() if hasattr(entry, "is_dir") else False,
            })

        response = {
            "success": True,
            "directory": directory,
            "files": files,
            "count": len(files),
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


# =============================================================================
# AGENTFS KV STORE TOOLS
# =============================================================================


@tool("set_state", "Save state in KV store", {"key": str, "value": str})
async def set_state(args: dict[str, Any]) -> dict[str, Any]:
    """
    Save state in KV store.

    Args:
        key: Unique key (e.g., "user_preferences")
        value: Value to save (string or JSON)

    Returns:
        Confirmation
    """
    key = args["key"]
    value = args["value"]

    try:
        rag = await _get_rag()
        await rag.kv.set(key, value)

        response = {
            "success": True,
            "key": key,
            "size": len(value),
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("get_state", "Get state from KV store", {"key": str})
async def get_state(args: dict[str, Any]) -> dict[str, Any]:
    """
    Get state from KV store.

    Args:
        key: State key

    Returns:
        Stored value or error
    """
    key = args["key"]

    try:
        rag = await _get_rag()
        value = await rag.kv.get(key)

        if value is None:
            return {
                "content": [{"type": "text", "text": f"Chave '{key}' não encontrada"}],
                "is_error": True,
            }

        response = {
            "success": True,
            "key": key,
            "value": value,
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("list_states", "List keys in KV store", {"prefix": str})
async def list_states(args: dict[str, Any]) -> dict[str, Any]:
    """
    List keys in KV store.

    Args:
        prefix: Optional prefix to filter keys

    Returns:
        List of keys
    """
    prefix = args.get("prefix", "")

    try:
        rag = await _get_rag()
        keys = await rag.kv.list(prefix=prefix if prefix else None)

        response = {
            "success": True,
            "prefix": prefix or "(all)",
            "keys": keys,
            "count": len(keys),
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


# =============================================================================
# METRICS & HEALTH TOOLS
# =============================================================================


@tool("get_metrics", "Get RAG system metrics", {})
async def get_metrics(args: dict[str, Any]) -> dict[str, Any]:
    """
    Get RAG system metrics.

    Returns:
        Statistics including cache, queries, and errors
    """
    try:
        rag = await _get_rag()
        stats = await rag.stats()

        return {
            "content": [{"type": "text", "text": json.dumps(stats, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
            "is_error": True,
        }


@tool("get_health", "Get system health status", {})
async def get_health(args: dict[str, Any]) -> dict[str, Any]:
    """
    Get system health status.

    Returns:
        Health check with component status
    """
    try:
        rag = await _get_rag()
        stats = await rag.stats()

        response = {
            "status": "healthy",
            "session_id": _session_id,
            "documents": stats.get("documents", {}).get("total_documents", 0),
            "cache_enabled": stats.get("cache", {}).get("enabled", False),
        }

        return {
            "content": [{"type": "text", "text": json.dumps(response, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": json.dumps({"status": "unhealthy", "error": str(e)}, ensure_ascii=False)}],
            "is_error": True,
        }
