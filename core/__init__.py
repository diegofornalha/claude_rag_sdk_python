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
from .hybrid_search import BM25, HybridSearch, SearchResult
from .logger import get_conversation_id, get_request_id, logger, set_conversation_id, set_request_id
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
    # Logger
    "logger",
    "set_conversation_id",
    "set_request_id",
    "get_conversation_id",
    "get_request_id",
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
