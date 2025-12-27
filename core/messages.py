# =============================================================================
# MESSAGES - Mensagens Padronizadas PT-BR
# =============================================================================
# Centraliza todas as mensagens do sistema para consistência e i18n
# =============================================================================

from enum import Enum
from typing import Any


class LogTag(str, Enum):
    """Tags para categorização de logs."""

    # Sistema
    SYSTEM = "SYSTEM"
    INIT = "INIT"
    SHUTDOWN = "SHUTDOWN"
    CONFIG = "CONFIG"

    # Operações
    SEARCH = "SEARCH"
    INGEST = "INGEST"
    CHAT = "CHAT"
    STREAM = "STREAM"
    RAG = "RAG"

    # Infraestrutura
    DB = "DB"
    CACHE = "CACHE"
    AUTH = "AUTH"
    RATE_LIMIT = "RATE"

    # Status
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    DEBUG = "DEBUG"

    # Sessão
    SESSION = "SESSION"
    JSONL = "JSONL"

    # MCP
    MCP = "MCP"
    TOOL = "TOOL"


class Messages:
    """Mensagens padronizadas em PT-BR."""

    # === Sistema ===
    SYSTEM_STARTED = "Sistema iniciado"
    SYSTEM_SHUTDOWN = "Sistema encerrado"
    CONFIG_LOADED = "Configuração carregada"
    CONFIG_RELOADED = "Configuração recarregada"

    # === Autenticação ===
    AUTH_SUCCESS = "Autenticação bem-sucedida"
    AUTH_FAILED = "Autenticação falhou"
    AUTH_KEY_INVALID = "API key inválida"
    AUTH_KEY_EXPIRED = "API key expirada"
    AUTH_KEY_MASKED = "API key mascarada"

    # === Busca (Search) ===
    SEARCH_STARTED = "Busca iniciada"
    SEARCH_COMPLETED = "Busca concluída"
    SEARCH_FAILED = "Busca falhou"
    SEARCH_NO_RESULTS = "Nenhum resultado encontrado"
    SEARCH_RESULTS_FOUND = "Resultados encontrados: {count}"

    # === Ingestão ===
    INGEST_STARTED = "Ingestão iniciada"
    INGEST_COMPLETED = "Ingestão concluída"
    INGEST_FAILED = "Ingestão falhou"
    INGEST_DOC_ADDED = "Documento adicionado"
    INGEST_DOC_UPDATED = "Documento atualizado"
    INGEST_DOC_SKIPPED = "Documento ignorado (já existe)"

    # === Chat ===
    CHAT_STARTED = "Chat iniciado"
    CHAT_COMPLETED = "Chat concluído"
    CHAT_FAILED = "Chat falhou"
    CHAT_STREAM_STARTED = "Stream iniciado"
    CHAT_STREAM_COMPLETED = "Stream concluído"
    CHAT_RAG_CONTEXT = "Contexto RAG incluído: {chars} caracteres"

    # === Sessão ===
    SESSION_CREATED = "Sessão criada"
    SESSION_LOADED = "Sessão carregada"
    SESSION_SAVED = "Sessão salva"
    SESSION_DELETED = "Sessão deletada"
    SESSION_NOT_FOUND = "Sessão não encontrada"
    SESSION_INVALID_ID = "ID de sessão inválido"
    SESSION_FAVORITE_ADDED = "Sessão adicionada aos favoritos"
    SESSION_FAVORITE_REMOVED = "Sessão removida dos favoritos"
    SESSION_RENAMED = "Sessão renomeada para: {name}"
    SESSION_TEMP_REMOVED = "Sessão temporária removida"

    # === JSONL ===
    JSONL_APPENDED = "Mensagens salvas em {session_id}.jsonl"
    JSONL_CREATED = "Arquivo JSONL criado para sessão"
    JSONL_FAILED = "Falha ao salvar JSONL"

    # === Database ===
    DB_CONNECTED = "Conexão com banco estabelecida"
    DB_DISCONNECTED = "Conexão com banco encerrada"
    DB_ERROR = "Erro no banco de dados"
    DB_NOT_FOUND = "Banco de dados não encontrado"
    DB_RESET = "Banco de dados resetado"

    # === Cache ===
    CACHE_HIT = "Cache hit"
    CACHE_MISS = "Cache miss"
    CACHE_EXPIRED = "Cache expirado"
    CACHE_CLEARED = "Cache limpo"

    # === Rate Limiting ===
    RATE_LIMIT_EXCEEDED = "Rate limit excedido"
    RATE_LIMIT_WARNING = "Próximo do rate limit"

    # === Erros ===
    ERROR_GENERIC = "Erro inesperado"
    ERROR_VALIDATION = "Erro de validação"
    ERROR_NOT_FOUND = "Recurso não encontrado"
    ERROR_PERMISSION = "Permissão negada"
    ERROR_TIMEOUT = "Timeout excedido"
    ERROR_CONNECTION = "Erro de conexão"

    # === MCP ===
    MCP_ADAPTER_ENABLED = "Adapter MCP habilitado: {name}"
    MCP_ADAPTER_DISABLED = "Adapter MCP desabilitado: {name}"
    MCP_TOOL_EXECUTED = "Ferramenta MCP executada: {tool}"
    MCP_TOOL_FAILED = "Ferramenta MCP falhou: {tool}"

    # === RAG ===
    RAG_CONTEXT_FOUND = "Contexto RAG encontrado"
    RAG_CONTEXT_EMPTY = "Contexto RAG vazio"
    RAG_SEARCH_FAILED = "Busca RAG falhou"
    RAG_STATS_RETRIEVED = "Estatísticas RAG obtidas"

    # === Embeddings ===
    EMBEDDING_STARTED = "Geração de embedding iniciada"
    EMBEDDING_COMPLETED = "Embedding gerado: {dims} dimensões"
    EMBEDDING_CACHED = "Embedding obtido do cache"
    EMBEDDING_FAILED = "Falha ao gerar embedding"

    # === Outputs ===
    OUTPUT_CREATED = "Arquivo de saída criado: {filename}"
    OUTPUT_DELETED = "Arquivo de saída deletado: {filename}"
    OUTPUT_NOT_FOUND = "Arquivo de saída não encontrado"
    OUTPUT_LIST_RETRIEVED = "Lista de arquivos obtida: {count} arquivos"

    # === Segurança ===
    SECURITY_PATH_TRAVERSAL = "Tentativa de path traversal bloqueada"
    SECURITY_PROMPT_INJECTION = "Possível prompt injection detectado"
    SECURITY_INVALID_INPUT = "Entrada inválida rejeitada"

    # === Circuit Breaker ===
    CIRCUIT_OPEN = "Circuit breaker aberto: {service}"
    CIRCUIT_CLOSED = "Circuit breaker fechado: {service}"
    CIRCUIT_HALF_OPEN = "Circuit breaker em half-open: {service}"

    # === Reranking ===
    RERANK_STARTED = "Reranking iniciado com {count} resultados"
    RERANK_COMPLETED = "Reranking concluído"
    RERANK_SKIPPED = "Reranking ignorado (poucos resultados)"

    @staticmethod
    def format(message: str, **kwargs: Any) -> str:
        """Formata uma mensagem com parâmetros."""
        try:
            return message.format(**kwargs)
        except KeyError:
            return message


def log_msg(tag: LogTag, message: str, **kwargs: Any) -> str:
    """Formata mensagem com tag para logging.

    Args:
        tag: Tag de categorização
        message: Mensagem base
        **kwargs: Parâmetros para formatação

    Returns:
        Mensagem formatada com tag

    Exemplo:
        log_msg(LogTag.CHAT, Messages.CHAT_STARTED)
        # "[CHAT] Chat iniciado"

        log_msg(LogTag.SEARCH, Messages.SEARCH_RESULTS_FOUND, count=5)
        # "[SEARCH] Resultados encontrados: 5"
    """
    formatted = Messages.format(message, **kwargs) if kwargs else message
    return f"[{tag.value}] {formatted}"


# Aliases para uso rápido
M = Messages
L = LogTag
