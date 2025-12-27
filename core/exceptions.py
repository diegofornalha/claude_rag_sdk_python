# =============================================================================
# EXCEPTIONS - Hierarquia de Exceções Customizadas
# =============================================================================
# Exceções específicas do domínio para tratamento de erros preciso
# =============================================================================

from typing import Any, Optional


class RAGException(Exception):
    """Exceção base para todas as exceções do sistema RAG.

    Atributos:
        message: Mensagem de erro descritiva
        code: Código de erro interno (para logging/debugging)
        details: Detalhes adicionais do erro
        http_status: Status HTTP sugerido para a resposta
    """

    http_status: int = 500
    default_message: str = "Erro interno do sistema"

    def __init__(
        self,
        message: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message or self.default_message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Converte exceção para dicionário (útil para respostas HTTP)."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Exceções de Database
# =============================================================================


class DatabaseError(RAGException):
    """Erro relacionado ao banco de dados."""

    http_status = 500
    default_message = "Erro no banco de dados"


class DatabaseConnectionError(DatabaseError):
    """Erro de conexão com o banco."""

    default_message = "Não foi possível conectar ao banco de dados"


class DatabaseNotFoundError(DatabaseError):
    """Banco de dados não encontrado."""

    http_status = 404
    default_message = "Banco de dados não encontrado"


class DatabaseQueryError(DatabaseError):
    """Erro na execução de query."""

    default_message = "Erro ao executar query no banco de dados"


# =============================================================================
# Exceções de Embedding
# =============================================================================


class EmbeddingError(RAGException):
    """Erro relacionado a embeddings."""

    http_status = 500
    default_message = "Erro ao gerar embeddings"


class EmbeddingModelError(EmbeddingError):
    """Erro no modelo de embedding."""

    default_message = "Erro ao carregar modelo de embedding"


class EmbeddingDimensionError(EmbeddingError):
    """Dimensões de embedding incompatíveis."""

    http_status = 400
    default_message = "Dimensões de embedding incompatíveis"


# =============================================================================
# Exceções de Search
# =============================================================================


class SearchError(RAGException):
    """Erro relacionado à busca."""

    http_status = 500
    default_message = "Erro na busca"


class SearchTimeoutError(SearchError):
    """Timeout na busca."""

    http_status = 504
    default_message = "Timeout na busca"


class SearchNoResultsError(SearchError):
    """Nenhum resultado encontrado."""

    http_status = 404
    default_message = "Nenhum resultado encontrado"


# =============================================================================
# Exceções de Ingestão
# =============================================================================


class IngestError(RAGException):
    """Erro relacionado à ingestão de documentos."""

    http_status = 500
    default_message = "Erro na ingestão de documentos"


class IngestFileError(IngestError):
    """Erro ao processar arquivo."""

    http_status = 400
    default_message = "Erro ao processar arquivo"


class IngestUnsupportedFormatError(IngestError):
    """Formato de arquivo não suportado."""

    http_status = 415
    default_message = "Formato de arquivo não suportado"


class IngestDuplicateError(IngestError):
    """Documento já existe."""

    http_status = 409
    default_message = "Documento já existe na base"


# =============================================================================
# Exceções de Autenticação
# =============================================================================


class AuthenticationError(RAGException):
    """Erro de autenticação."""

    http_status = 401
    default_message = "Autenticação falhou"


class InvalidAPIKeyError(AuthenticationError):
    """API key inválida."""

    default_message = "API key inválida"


class ExpiredAPIKeyError(AuthenticationError):
    """API key expirada."""

    default_message = "API key expirada"


class MissingAPIKeyError(AuthenticationError):
    """API key não fornecida."""

    default_message = "API key não fornecida"


# =============================================================================
# Exceções de Validação
# =============================================================================


class ValidationError(RAGException):
    """Erro de validação de dados."""

    http_status = 400
    default_message = "Erro de validação"


class InvalidInputError(ValidationError):
    """Entrada inválida."""

    default_message = "Entrada inválida"


class MissingRequiredFieldError(ValidationError):
    """Campo obrigatório ausente."""

    default_message = "Campo obrigatório ausente"


class InvalidSessionIdError(ValidationError):
    """ID de sessão inválido."""

    default_message = "ID de sessão inválido"


# =============================================================================
# Exceções de Rate Limiting
# =============================================================================


class RateLimitError(RAGException):
    """Rate limit excedido."""

    http_status = 429
    default_message = "Rate limit excedido"


class RateLimitExceededError(RateLimitError):
    """Muitas requisições."""

    default_message = "Muitas requisições. Tente novamente mais tarde."


# =============================================================================
# Exceções de Sessão
# =============================================================================


class SessionError(RAGException):
    """Erro relacionado a sessões."""

    http_status = 500
    default_message = "Erro de sessão"


class SessionNotFoundError(SessionError):
    """Sessão não encontrada."""

    http_status = 404
    default_message = "Sessão não encontrada"


class SessionExpiredError(SessionError):
    """Sessão expirada."""

    http_status = 401
    default_message = "Sessão expirada"


# =============================================================================
# Exceções de LLM
# =============================================================================


class LLMError(RAGException):
    """Erro relacionado ao LLM."""

    http_status = 502
    default_message = "Erro na comunicação com o LLM"


class LLMTimeoutError(LLMError):
    """Timeout na resposta do LLM."""

    http_status = 504
    default_message = "Timeout na resposta do LLM"


class LLMRateLimitError(LLMError):
    """Rate limit do LLM."""

    http_status = 429
    default_message = "Rate limit do provedor LLM excedido"


class LLMContextLengthError(LLMError):
    """Contexto muito longo para o LLM."""

    http_status = 400
    default_message = "Contexto excede o limite do modelo"


# =============================================================================
# Exceções de Segurança
# =============================================================================


class SecurityError(RAGException):
    """Erro de segurança."""

    http_status = 403
    default_message = "Erro de segurança"


class PromptInjectionError(SecurityError):
    """Tentativa de prompt injection detectada."""

    default_message = "Conteúdo potencialmente malicioso detectado"


class PathTraversalError(SecurityError):
    """Tentativa de path traversal detectada."""

    default_message = "Tentativa de acesso não autorizado"


# =============================================================================
# Exceções de MCP
# =============================================================================


class MCPError(RAGException):
    """Erro relacionado ao MCP."""

    http_status = 500
    default_message = "Erro no servidor MCP"


class MCPAdapterError(MCPError):
    """Erro no adapter MCP."""

    default_message = "Erro no adapter MCP"


class MCPToolError(MCPError):
    """Erro na execução de ferramenta MCP."""

    default_message = "Erro na execução da ferramenta"


# =============================================================================
# Helpers
# =============================================================================


def raise_for_status(response_code: int, message: Optional[str] = None) -> None:
    """Lança exceção apropriada baseada no código HTTP.

    Args:
        response_code: Código HTTP
        message: Mensagem customizada

    Raises:
        RAGException correspondente ao código
    """
    exceptions_map = {
        400: ValidationError,
        401: AuthenticationError,
        403: SecurityError,
        404: SearchNoResultsError,
        409: IngestDuplicateError,
        415: IngestUnsupportedFormatError,
        429: RateLimitError,
        500: RAGException,
        502: LLMError,
        504: SearchTimeoutError,
    }

    exc_class = exceptions_map.get(response_code, RAGException)
    raise exc_class(message=message)
