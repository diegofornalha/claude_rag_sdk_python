# =============================================================================
# SDK HOOKS - Sistema Completo de Auditoria e Segurança para Claude Agent SDK
# =============================================================================
# Implementação completa com:
# - RBAC (Role-Based Access Control) para tools e documentos
# - Rate Limiting thread-safe por usuário/tool
# - AuditDatabase (SQLite) para persistência de eventos
# - HooksManager para gerenciar PreToolUse e PostToolUse
# =============================================================================

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# =============================================================================
# 1. DATA MODELS - Estruturas de dados para auditoria
# =============================================================================


class AuditEventType(str, Enum):
    """Tipos de eventos de auditoria."""

    TOOL_CALL = "tool_call"
    TOOL_BLOCKED = "tool_blocked"
    TOOL_SUCCESS = "tool_success"
    TOOL_ERROR = "tool_error"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMITED = "rate_limited"
    RBAC_VIOLATION = "rbac_violation"
    DOCUMENT_ACCESS = "document_access"
    SECURITY_BLOCKED = "security_blocked"


class AuditSeverity(str, Enum):
    """Níveis de severidade."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditRecord:
    """Registro de evento de auditoria."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.TOOL_CALL
    severity: AuditSeverity = AuditSeverity.INFO
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: str = ""
    session_id: str = ""
    tool_name: str = ""
    tool_parameters: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    blocked_reason: Optional[str] = None
    documents_accessed: int = 0
    documents_filtered: int = 0
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converter para dict para armazenagem."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data


@dataclass
class UserContext:
    """Contexto de usuário com permissões."""

    user_id: str
    roles: list[str] = field(default_factory=list)  # ['admin', 'analyst']
    tags: list[str] = field(default_factory=list)  # ['finance', 'public']
    rate_limit_calls: int = 100
    rate_limit_window_seconds: int = 3600


# =============================================================================
# 2. AUDIT DATABASE - Persistência em SQLite
# =============================================================================


class AuditDatabase:
    """Gerencia persistência de eventos de auditoria em SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.cwd() / ".agentfs" / "audit.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Criar tabelas se não existirem."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'info',
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_parameters TEXT,
                    result TEXT,
                    blocked_reason TEXT,
                    documents_accessed INTEGER DEFAULT 0,
                    documents_filtered INTEGER DEFAULT 0,
                    duration_ms REAL DEFAULT 0.0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_name ON audit_events(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON audit_events(session_id)")
            conn.commit()

    def log_event(self, record: AuditRecord):
        """Registrar evento de auditoria."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, event_type, severity, timestamp, user_id, session_id,
                        tool_name, tool_parameters, result, blocked_reason,
                        documents_accessed, documents_filtered, duration_ms, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.event_id,
                        record.event_type.value,
                        record.severity.value,
                        record.timestamp,
                        record.user_id,
                        record.session_id,
                        record.tool_name,
                        json.dumps(record.tool_parameters),
                        json.dumps(record.result),
                        record.blocked_reason,
                        record.documents_accessed,
                        record.documents_filtered,
                        record.duration_ms,
                        json.dumps(record.metadata),
                    ),
                )
                conn.commit()

    def query_events(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        tool_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Consultar eventos de auditoria."""
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        if tool_name:
            query += " AND tool_name = ?"
            params.append(tool_name)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self, session_id: Optional[str] = None) -> dict:
        """Obter estatísticas de auditoria."""
        base_query = "SELECT COUNT(*) as count, event_type FROM audit_events"
        params = []

        if session_id:
            base_query += " WHERE session_id = ?"
            params.append(session_id)

        base_query += " GROUP BY event_type"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(base_query, params)
            stats = {row[1]: row[0] for row in cursor.fetchall()}

        return {
            "total_events": sum(stats.values()),
            "by_type": stats,
            "blocked_count": stats.get(AuditEventType.TOOL_BLOCKED.value, 0)
            + stats.get(AuditEventType.RBAC_VIOLATION.value, 0)
            + stats.get(AuditEventType.RATE_LIMITED.value, 0),
        }


# =============================================================================
# 3. RBAC MANAGER - Controle de Acesso
# =============================================================================


class RBACManager:
    """Gerencia controle de acesso baseado em roles."""

    # Mapa de tool -> roles necessários
    TOOL_PERMISSIONS = {
        "search_legal_docs": ["admin", "analyst", "auditor"],
        "search_financial_docs": ["admin", "analyst"],
        "search_public_docs": ["admin", "analyst", "viewer"],
        "modify_document": ["admin"],
        "export_data": ["admin", "analyst"],
        "view_audit_logs": ["admin", "auditor"],
        "Write": ["admin", "analyst"],  # Claude Agent SDK tools
        "Edit": ["admin", "analyst"],
        "Bash": ["admin"],  # Comandos bash restritos
    }

    # Mapa de documento tag -> roles que podem acessar
    DOCUMENT_ACCESS_CONTROL = {
        "public": ["admin", "analyst", "viewer"],
        "financial": ["admin", "analyst"],
        "legal": ["admin", "analyst"],
        "sensitive": ["admin"],
        "confidential": ["admin"],
    }

    def can_access_tool(self, user: UserContext, tool_name: str) -> bool:
        """Verificar se user pode acessar uma tool."""
        required_roles = self.TOOL_PERMISSIONS.get(tool_name, [])
        if not required_roles:
            return True  # Tool pública

        return any(role in user.roles for role in required_roles)

    def can_access_document(self, user: UserContext, doc_tags: list[str]) -> bool:
        """Verificar se user pode acessar um documento por suas tags."""
        for tag in doc_tags:
            allowed_roles = self.DOCUMENT_ACCESS_CONTROL.get(tag, [])
            if allowed_roles and not any(role in user.roles for role in allowed_roles):
                return False
        return True

    def get_accessible_document_tags(self, user: UserContext) -> list[str]:
        """Retornar todas as tags de documento que o user pode acessar."""
        accessible = []
        for tag, roles in self.DOCUMENT_ACCESS_CONTROL.items():
            if any(role in user.roles for role in roles):
                accessible.append(tag)
        return accessible


# =============================================================================
# 4. RATE LIMIT MANAGER - Proteção contra Abuso
# =============================================================================


class RateLimitManager:
    """Gerencia rate limiting por user e tool."""

    def __init__(self):
        self.call_history: dict[str, dict[str, list[float]]] = {}
        self.lock = threading.Lock()

    def check_rate_limit(
        self, user_id: str, tool_name: str, max_calls: int = 100, window_seconds: int = 3600
    ) -> tuple[bool, int]:
        """
        Verificar se user atingiu rate limit para uma tool.

        Returns:
            Tuple of (allowed: bool, remaining_calls: int)
        """
        with self.lock:
            now = time.time()

            # Inicializar estrutura se necessário
            if user_id not in self.call_history:
                self.call_history[user_id] = {}
            if tool_name not in self.call_history[user_id]:
                self.call_history[user_id][tool_name] = []

            # Limpar timestamps antigos
            cutoff_time = now - window_seconds
            self.call_history[user_id][tool_name] = [
                ts for ts in self.call_history[user_id][tool_name] if ts > cutoff_time
            ]

            # Verificar limite
            call_count = len(self.call_history[user_id][tool_name])
            remaining = max_calls - call_count

            if call_count >= max_calls:
                return False, 0

            # Registrar nova chamada
            self.call_history[user_id][tool_name].append(now)
            return True, remaining - 1

    def get_usage(self, user_id: str) -> dict[str, int]:
        """Obter uso atual de rate limit por tool."""
        with self.lock:
            if user_id not in self.call_history:
                return {}
            return {tool: len(calls) for tool, calls in self.call_history[user_id].items()}


# =============================================================================
# 5. HOOKS MANAGER - Gerencia PreToolUse e PostToolUse
# =============================================================================


class HooksManager:
    """
    Gerencia os hooks de pré e pós tool execution.

    Integra RBAC, Rate Limiting e Auditoria em um único manager.
    """

    # Tools bloqueadas por segurança (sempre negadas)
    BLOCKED_TOOLS = {"rm", "sudo", "chmod", "chown", "mkfs", "dd", "format", "shutdown", "reboot"}

    def __init__(
        self,
        rbac: Optional[RBACManager] = None,
        rate_limiter: Optional[RateLimitManager] = None,
        audit_db: Optional[AuditDatabase] = None,
        enable_rbac: bool = False,  # Desativado por padrão (sem autenticação)
        enable_rate_limit: bool = True,
        max_calls_per_hour: int = 100,
    ):
        self.rbac = rbac or RBACManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.audit_db = audit_db or AuditDatabase()
        self.enable_rbac = enable_rbac
        self.enable_rate_limit = enable_rate_limit
        self.max_calls_per_hour = max_calls_per_hour

        # Armazenar timestamps para calcular duração
        self._tool_start_times: dict[str, float] = {}

        # Contexto de usuário padrão (quando RBAC desativado)
        self._default_user = UserContext(
            user_id="default",
            roles=["admin"],  # Acesso total quando RBAC desativado
            tags=["public", "financial", "legal", "sensitive"],
        )

    async def pre_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: Optional[str],
        context: Any,
    ) -> dict[str, Any]:
        """
        Hook executado ANTES de cada tool call.

        Responsabilidades:
        - Bloquear tools perigosas
        - Validar RBAC (se habilitado)
        - Verificar rate limiting
        - Registrar tentativa de acesso

        Args:
            input_data: Dados do tool use (tool_name, tool_input)
            tool_use_id: ID único do tool use
            context: Contexto do hook (HookContext)

        Returns:
            {} para permitir, ou {"hookSpecificOutput": {...}} para bloquear
        """
        tool_name = input_data.get("tool_name", input_data.get("name", "unknown"))
        tool_input = input_data.get("tool_input", input_data.get("input", {}))
        hook_event_name = input_data.get("hook_event_name", "PreToolUse")

        # Registrar timestamp de início
        if tool_use_id:
            self._tool_start_times[tool_use_id] = time.time()

        # Extrair contexto de usuário
        user = self._get_user_context(context)
        session_id = self._get_session_id(context)

        # Log no console
        print(f"[HOOK:PreToolUse] Tool: {tool_name} | User: {user.user_id} | ID: {tool_use_id}")

        # 1. Verificar tools bloqueadas por segurança
        if self._is_blocked_tool(tool_name):
            record = AuditRecord(
                event_type=AuditEventType.SECURITY_BLOCKED,
                severity=AuditSeverity.WARNING,
                user_id=user.user_id,
                session_id=session_id,
                tool_name=tool_name,
                tool_parameters=self._truncate_params(tool_input),
                blocked_reason=f"Tool '{tool_name}' is blocked for security reasons",
            )
            self.audit_db.log_event(record)
            print(f"[HOOK:PreToolUse] BLOCKED (security): {tool_name}")

            return {
                "hookSpecificOutput": {
                    "hookEventName": hook_event_name,
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Tool '{tool_name}' is blocked for security",
                }
            }

        # 2. Validar RBAC (se habilitado)
        if self.enable_rbac and not self.rbac.can_access_tool(user, tool_name):
            record = AuditRecord(
                event_type=AuditEventType.RBAC_VIOLATION,
                severity=AuditSeverity.WARNING,
                user_id=user.user_id,
                session_id=session_id,
                tool_name=tool_name,
                tool_parameters=self._truncate_params(tool_input),
                blocked_reason=f"User lacks required roles for tool '{tool_name}'",
            )
            self.audit_db.log_event(record)
            print(f"[HOOK:PreToolUse] BLOCKED (RBAC): {tool_name}")

            return {
                "hookSpecificOutput": {
                    "hookEventName": hook_event_name,
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Access denied to tool '{tool_name}'",
                }
            }

        # 3. Verificar rate limiting
        if self.enable_rate_limit:
            allowed, remaining = self.rate_limiter.check_rate_limit(
                user.user_id, tool_name, self.max_calls_per_hour, 3600
            )
            if not allowed:
                record = AuditRecord(
                    event_type=AuditEventType.RATE_LIMITED,
                    severity=AuditSeverity.WARNING,
                    user_id=user.user_id,
                    session_id=session_id,
                    tool_name=tool_name,
                    tool_parameters=self._truncate_params(tool_input),
                    blocked_reason=f"Rate limit exceeded for tool '{tool_name}'",
                )
                self.audit_db.log_event(record)
                print(f"[HOOK:PreToolUse] BLOCKED (rate limit): {tool_name}")

                return {
                    "hookSpecificOutput": {
                        "hookEventName": hook_event_name,
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Rate limit exceeded for '{tool_name}'",
                    }
                }

        # 4. Registrar tentativa (permitida)
        record = AuditRecord(
            event_type=AuditEventType.TOOL_CALL,
            severity=AuditSeverity.INFO,
            user_id=user.user_id,
            session_id=session_id,
            tool_name=tool_name,
            tool_parameters=self._truncate_params(tool_input),
            metadata={"status": "attempting", "tool_use_id": tool_use_id},
        )
        self.audit_db.log_event(record)

        # Permitir execução
        return {}

    async def post_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: Optional[str],
        context: Any,
    ) -> dict[str, Any]:
        """
        Hook executado DEPOIS de cada tool call.

        Responsabilidades:
        - Calcular duração da execução
        - Registrar sucesso ou erro
        - Filtrar documentos por RBAC (se aplicável)

        Args:
            input_data: Dados do resultado (tool_name, tool_result, is_error)
            tool_use_id: ID único do tool use
            context: Contexto do hook

        Returns:
            {} para não modificar resultado
        """
        tool_name = input_data.get("tool_name", input_data.get("name", "unknown"))
        tool_result = input_data.get("tool_result", input_data.get("result", {}))
        is_error = input_data.get("is_error", False)

        # Calcular duração
        duration_ms = None
        if tool_use_id and tool_use_id in self._tool_start_times:
            start_time = self._tool_start_times.pop(tool_use_id)
            duration_ms = (time.time() - start_time) * 1000

        # Extrair contexto
        user = self._get_user_context(context)
        session_id = self._get_session_id(context)

        # Log no console
        status = "ERROR" if is_error else "OK"
        duration_str = f" | Duration: {duration_ms:.2f}ms" if duration_ms else ""
        print(f"[HOOK:PostToolUse] Tool: {tool_name} | Status: {status}{duration_str}")

        # Registrar evento
        if is_error:
            record = AuditRecord(
                event_type=AuditEventType.TOOL_ERROR,
                severity=AuditSeverity.ERROR,
                user_id=user.user_id,
                session_id=session_id,
                tool_name=tool_name,
                result=self._truncate_result(tool_result),
                duration_ms=duration_ms or 0.0,
                metadata={"tool_use_id": tool_use_id, "is_error": True},
            )
        else:
            record = AuditRecord(
                event_type=AuditEventType.TOOL_SUCCESS,
                severity=AuditSeverity.INFO,
                user_id=user.user_id,
                session_id=session_id,
                tool_name=tool_name,
                result=self._truncate_result(tool_result),
                duration_ms=duration_ms or 0.0,
                metadata={"tool_use_id": tool_use_id, "is_error": False},
            )

        self.audit_db.log_event(record)

        # Não modificar resultado
        return {}

    def _is_blocked_tool(self, tool_name: str) -> bool:
        """Verificar se tool está na blocklist."""
        tool_lower = tool_name.lower()
        return any(blocked in tool_lower for blocked in self.BLOCKED_TOOLS)

    def _get_user_context(self, context: Any) -> UserContext:
        """Extrair contexto de usuário do hook context."""
        if context is None:
            return self._default_user

        # Tentar extrair de diferentes formatos de contexto
        if hasattr(context, "user_id"):
            return UserContext(
                user_id=getattr(context, "user_id", "unknown"),
                roles=getattr(context, "roles", ["admin"]),
                tags=getattr(context, "tags", []),
            )

        if isinstance(context, dict):
            return UserContext(
                user_id=context.get("user_id", "unknown"),
                roles=context.get("roles", ["admin"]),
                tags=context.get("tags", []),
            )

        return self._default_user

    def _get_session_id(self, context: Any) -> str:
        """Extrair session_id do contexto."""
        if context is None:
            return "unknown"

        if hasattr(context, "session_id"):
            return getattr(context, "session_id", "unknown")

        if isinstance(context, dict):
            return context.get("session_id", "unknown")

        return "unknown"

    def _truncate_params(self, params: Any, max_len: int = 500) -> dict:
        """Truncar parâmetros para não sobrecarregar logs."""
        if params is None:
            return {}

        if isinstance(params, str):
            if len(params) > max_len:
                return {"value": params[:max_len] + "...[TRUNCATED]"}
            return {"value": params}

        if isinstance(params, dict):
            return {
                k: (str(v)[:max_len] + "..." if len(str(v)) > max_len else v)
                for k, v in list(params.items())[:10]
            }

        return {"value": str(params)[:max_len]}

    def _truncate_result(self, result: Any, max_len: int = 500) -> dict:
        """Truncar resultado para não sobrecarregar logs."""
        if result is None:
            return {}

        if isinstance(result, str):
            if len(result) > max_len:
                return {"content": result[:max_len] + "...[TRUNCATED]"}
            return {"content": result}

        if isinstance(result, dict):
            return {
                k: (str(v)[:max_len] + "..." if isinstance(v, str) and len(str(v)) > max_len else v)
                for k, v in list(result.items())[:10]
            }

        if isinstance(result, list):
            return {"items": len(result), "sample": result[:3] if len(result) > 3 else result}

        return {"value": str(result)[:max_len]}


# =============================================================================
# 6. GLOBAL INSTANCES - Instâncias compartilhadas
# =============================================================================

# Instância global do HooksManager
_hooks_manager: Optional[HooksManager] = None


def get_hooks_manager() -> HooksManager:
    """Obter instância global do HooksManager."""
    global _hooks_manager
    if _hooks_manager is None:
        _hooks_manager = HooksManager()
    return _hooks_manager


def get_audit_database() -> AuditDatabase:
    """Obter instância do AuditDatabase."""
    return get_hooks_manager().audit_db


# =============================================================================
# 7. SDK HOOKS CONFIG - Configuração para ClaudeAgentOptions
# =============================================================================


async def standalone_pre_tool_use(
    input_data: dict[str, Any],
    tool_use_id: Optional[str],
    context: Any,
) -> dict[str, Any]:
    """Standalone wrapper for pre_tool_use hook."""
    print(f"[HOOK] >>> PreToolUse CALLED! tool={input_data.get('tool_name', 'unknown')}")
    manager = get_hooks_manager()
    return await manager.pre_tool_use(input_data, tool_use_id, context)


async def standalone_post_tool_use(
    input_data: dict[str, Any],
    tool_use_id: Optional[str],
    context: Any,
) -> dict[str, Any]:
    """Standalone wrapper for post_tool_use hook."""
    print(f"[HOOK] >>> PostToolUse CALLED! tool={input_data.get('tool_name', 'unknown')}")
    manager = get_hooks_manager()
    return await manager.post_tool_use(input_data, tool_use_id, context)


def get_sdk_hooks_config() -> dict:
    """
    Retorna configuração de hooks para ClaudeAgentOptions.

    Uso:
        options = ClaudeAgentOptions(
            ...,
            hooks=get_sdk_hooks_config()
        )

    Returns:
        dict compatível com ClaudeAgentOptions.hooks

    Formato oficial do Claude Agent SDK:
        - Chaves em PascalCase: PreToolUse, PostToolUse
        - Valor: lista de dicts com 'hooks' (lista de funções async)
    """
    return {
        "PreToolUse": [
            {
                "hooks": [standalone_pre_tool_use],
            }
        ],
        "PostToolUse": [
            {
                "hooks": [standalone_post_tool_use],
            }
        ],
    }


# =============================================================================
# 8. LEGACY COMPATIBILITY - Manter compatibilidade com audit.py
# =============================================================================


class AuditLogger:
    """Logger de auditoria para compatibilidade com código legado."""

    def __init__(self):
        self._db = get_audit_database()

    def log_tool_call(
        self,
        tool_name: str,
        inputs: Any = None,
        outputs: Any = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[dict] = None,
    ):
        """Log de chamada de tool."""
        record = AuditRecord(
            event_type=AuditEventType.TOOL_CALL,
            tool_name=tool_name,
            tool_parameters=inputs if isinstance(inputs, dict) else {"value": str(inputs)},
            result=outputs if isinstance(outputs, dict) else {"value": str(outputs)},
            duration_ms=duration_ms or 0.0,
            metadata=metadata or {},
        )
        self._db.log_event(record)

    def log_blocked_attempt(
        self,
        tool_name: str,
        reason: str,
        inputs: Any = None,
        metadata: Optional[dict] = None,
    ):
        """Log de tentativa bloqueada."""
        record = AuditRecord(
            event_type=AuditEventType.TOOL_BLOCKED,
            severity=AuditSeverity.WARNING,
            tool_name=tool_name,
            tool_parameters=inputs if isinstance(inputs, dict) else {"value": str(inputs)},
            blocked_reason=reason,
            metadata=metadata or {},
        )
        self._db.log_event(record)

    def log_error(
        self,
        error_type: str,
        error_message: str,
        tool_name: str = "",
        metadata: Optional[dict] = None,
    ):
        """Log de erro."""
        record = AuditRecord(
            event_type=AuditEventType.TOOL_ERROR,
            severity=AuditSeverity.ERROR,
            tool_name=tool_name,
            result={"error_type": error_type, "error_message": error_message},
            metadata=metadata or {},
        )
        self._db.log_event(record)


# Instância global do AuditLogger (compatibilidade)
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Obter instância global do AuditLogger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
