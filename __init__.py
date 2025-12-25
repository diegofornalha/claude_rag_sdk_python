"""Claude RAG SDK - RAG capabilities for Claude with AgentFS integration.

A powerful SDK for building RAG (Retrieval-Augmented Generation) applications
with Claude, built on top of AgentFS for state management and audit trails.

Example:
    >>> from claude_rag_sdk import ClaudeRAG, ClaudeRAGOptions
    >>>
    >>> async def main():
    ...     async with await ClaudeRAG.open(ClaudeRAGOptions(id='my-agent')) as rag:
    ...         # Add documents
    ...         await rag.ingest.add_document('manual.pdf')
    ...
    ...         # Search
    ...         results = await rag.search('What is RAG?')
    ...
    ...         # Ask questions
    ...         response = await rag.query('Explain RAG in detail')
    ...         print(response.answer)
    ...
    ...         # Use AgentFS features
    ...         await rag.kv.set('last_query', 'What is RAG?')
    ...         await rag.fs.write_file('/output/report.txt', response.answer)
"""

from .agent import AgentEngine, AgentResponse, SimpleAgent, StreamChunk
from .ingest import Document, IngestEngine, IngestResult
from .options import AgentModel, ChunkingStrategy, ClaudeRAGOptions, EmbeddingModel
from .rag import ClaudeRAG
from .search import HybridSearchResult, SearchEngine, SearchResult

__version__ = "0.1.0"

__all__ = [
    # Main class
    "ClaudeRAG",
    "ClaudeRAGOptions",
    # Options/Config
    "EmbeddingModel",
    "ChunkingStrategy",
    "AgentModel",
    # Search
    "SearchEngine",
    "SearchResult",
    "HybridSearchResult",
    # Ingest
    "IngestEngine",
    "IngestResult",
    "Document",
    # Agent
    "AgentEngine",
    "AgentResponse",
    "StreamChunk",
    "SimpleAgent",
]
