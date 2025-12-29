# Core modules for RAG Agent
from .auth import (
    APIKey,
    APIKeyManager,
    AuthResult,
    AuthScope,
    authenticate,
    extract_api_key,
    get_key_manager,
)
from .cache import EmbeddingCache, LRUCache, ResponseCache, get_embedding_cache, get_response_cache
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
    circuit_breaker,
    get_or_create_circuit_breaker,
)
from .config import ChunkingStrategy, EmbeddingModel, RAGConfig, get_config, reload_config
from .database import ConnectionPool, get_audit_pool, get_rag_pool, reset_pools
from .exceptions import (
    AuthenticationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseNotFoundError,
    EmbeddingError,
    IngestError,
    IngestFileError,
    InvalidAPIKeyError,
    InvalidInputError,
    InvalidSessionIdError,
    LLMError,
    LLMTimeoutError,
    MCPError,
    PathTraversalError,
    PromptInjectionError,
    RAGException,
    RateLimitError,
    SearchError,
    SearchTimeoutError,
    SecurityError,
    SessionError,
    SessionNotFoundError,
    ValidationError,
    raise_for_status,
)
from .hybrid_search import BM25, HybridSearch, SearchResult
from .logger import (
    get_conversation_id,
    get_logger,
    get_request_id,
    logger,
    set_conversation_id,
    set_request_id,
)
from .messages import LogTag, Messages, log_msg
from .prompt_guard import (
    PromptGuard,
    ScanResult,
    ThreatLevel,
    get_prompt_guard,
    is_safe_prompt,
    scan_prompt,
)
from .rate_limiter import (
    RateLimitResult,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    check_rate_limit,
    get_rate_limiter,
)
from .rbac import RBACFilter, Role, User, get_current_user, get_rbac_filter, set_current_user
from .reranker import CrossEncoderReranker, LightweightReranker, RerankResult, create_reranker
from .security import CORSConfig, CORSMiddleware, SecurityHeaders, get_security_headers

__all__ = [
    # Config
    "EmbeddingModel",
    "ChunkingStrategy",
    "RAGConfig",
    "get_config",
    "reload_config",
    # Database
    "ConnectionPool",
    "get_rag_pool",
    "get_audit_pool",
    "reset_pools",
    # Exceptions
    "RAGException",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseNotFoundError",
    "EmbeddingError",
    "SearchError",
    "SearchTimeoutError",
    "IngestError",
    "IngestFileError",
    "AuthenticationError",
    "InvalidAPIKeyError",
    "ValidationError",
    "InvalidInputError",
    "InvalidSessionIdError",
    "RateLimitError",
    "SessionError",
    "SessionNotFoundError",
    "LLMError",
    "LLMTimeoutError",
    "SecurityError",
    "PromptInjectionError",
    "PathTraversalError",
    "MCPError",
    "raise_for_status",
    # Logger
    "logger",
    "get_logger",
    "set_conversation_id",
    "set_request_id",
    "get_conversation_id",
    "get_request_id",
    # Messages
    "LogTag",
    "Messages",
    "log_msg",
    # RBAC
    "User",
    "Role",
    "RBACFilter",
    "set_current_user",
    "get_current_user",
    "get_rbac_filter",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerError",
    "circuit_breaker",
    "get_or_create_circuit_breaker",
    # Cache
    "LRUCache",
    "EmbeddingCache",
    "ResponseCache",
    "get_embedding_cache",
    "get_response_cache",
    # Hybrid Search
    "HybridSearch",
    "BM25",
    "SearchResult",
    # Reranker
    "CrossEncoderReranker",
    "LightweightReranker",
    "create_reranker",
    "RerankResult",
    # Security
    "CORSMiddleware",
    "CORSConfig",
    "SecurityHeaders",
    "get_security_headers",
    # Rate Limiter
    "SlidingWindowRateLimiter",
    "TokenBucketRateLimiter",
    "RateLimitResult",
    "get_rate_limiter",
    "check_rate_limit",
    # Prompt Guard
    "PromptGuard",
    "ThreatLevel",
    "ScanResult",
    "get_prompt_guard",
    "scan_prompt",
    "is_safe_prompt",
    # Auth
    "APIKeyManager",
    "APIKey",
    "AuthScope",
    "AuthResult",
    "get_key_manager",
    "authenticate",
    "extract_api_key",
]
