# =============================================================================
# GUEST LIMITS - Sistema Flexível de Limites para Usuários Não-Autenticados
# =============================================================================
# Controla quantos prompts um usuário guest pode fazer antes de exigir signup
# Configurável via variáveis de ambiente para diferentes ambientes/tiers
# =============================================================================

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from claude_rag_sdk.core.logger import get_logger

logger = get_logger("guest_limits")


class GuestLimitAction(str, Enum):
    """Ação a ser tomada quando limite é atingido."""

    ALLOW = "allow"  # Permitir (ainda dentro do limite ou sem limite)
    SOFT_LIMIT = "soft_limit"  # Aviso (próximo do limite)
    REQUIRE_SIGNUP = "require_signup"  # Exigir signup
    BLOCK = "block"  # Bloquear completamente


@dataclass
class GuestLimitConfig:
    """Configuração de limites para guests.

    Attributes:
        prompts_limit: Número de prompts permitidos (0 = ilimitado)
        soft_limit_threshold: Prompt onde começa o aviso (ex: 1 = avisa no 1º)
        allow_in_dev: Se True, ignora limites em development
        grace_prompts: Prompts extras após limite (para UX suave)
    """

    prompts_limit: int = 1  # Default: 1 prompt grátis, depois signup
    soft_limit_threshold: int = 0  # 0 = sem aviso prévio
    allow_in_dev: bool = True  # Dev sempre liberado
    grace_prompts: int = 0  # Prompts extras após limite

    @classmethod
    def from_env(cls) -> "GuestLimitConfig":
        """Carrega configuração das variáveis de ambiente."""
        return cls(
            prompts_limit=int(os.getenv("GUEST_PROMPTS_LIMIT", "1")),
            soft_limit_threshold=int(os.getenv("GUEST_SOFT_LIMIT", "0")),
            allow_in_dev=os.getenv("GUEST_ALLOW_IN_DEV", "true").lower() == "true",
            grace_prompts=int(os.getenv("GUEST_GRACE_PROMPTS", "0")),
        )


@dataclass
class GuestCheckResult:
    """Resultado da verificação de limite de guest.

    Attributes:
        action: Ação recomendada
        prompt_count: Número atual de prompts
        prompts_remaining: Prompts restantes (-1 = ilimitado)
        message: Mensagem para o usuário
        can_continue: Se pode continuar sem autenticação
    """

    action: GuestLimitAction
    prompt_count: int
    prompts_remaining: int
    message: Optional[str] = None
    can_continue: bool = True

    def to_dict(self) -> dict:
        """Converte para dict para resposta da API."""
        return {
            "action": self.action.value,
            "prompt_count": self.prompt_count,
            "prompts_remaining": self.prompts_remaining,
            "message": self.message,
            "can_continue": self.can_continue,
        }


class GuestLimitManager:
    """Gerencia limites de uso para usuários guest (não autenticados).

    Uso:
        manager = GuestLimitManager()

        # Verificar antes de processar prompt
        result = await manager.check_and_increment(session_id, agentfs)

        if not result.can_continue:
            return {"error": "signup_required", "details": result.to_dict()}

        # Processar prompt normalmente...

    Configuração via .env:
        GUEST_PROMPTS_LIMIT=1      # 1 prompt grátis (0 = ilimitado)
        GUEST_SOFT_LIMIT=0         # Sem aviso prévio
        GUEST_ALLOW_IN_DEV=true    # Dev sempre liberado
        GUEST_GRACE_PROMPTS=0      # Sem prompts extras
        ENVIRONMENT=development    # Dev mode
    """

    def __init__(self, config: Optional[GuestLimitConfig] = None):
        self.config = config or GuestLimitConfig.from_env()
        self._environment = os.getenv("ENVIRONMENT", "development")

        logger.info(
            "Guest limits initialized",
            prompts_limit=self.config.prompts_limit,
            allow_in_dev=self.config.allow_in_dev,
            environment=self._environment,
        )

    def is_dev_mode(self) -> bool:
        """Verifica se está em modo desenvolvimento."""
        return self._environment.lower() in ("development", "dev", "local")

    def is_unlimited(self) -> bool:
        """Verifica se limites estão desativados."""
        # Dev mode com allow_in_dev = ilimitado
        if self.is_dev_mode() and self.config.allow_in_dev:
            return True
        # Limite 0 = ilimitado
        if self.config.prompts_limit == 0:
            return True
        return False

    async def get_prompt_count(self, agentfs) -> int:
        """Obtém contador de prompts da sessão."""
        try:
            count = await agentfs.kv.get("session:prompt_count")
            return int(count) if count is not None else 0
        except Exception:
            return 0

    async def increment_prompt_count(self, agentfs) -> int:
        """Incrementa e retorna novo contador de prompts."""
        current = await self.get_prompt_count(agentfs)
        new_count = current + 1
        await agentfs.kv.set("session:prompt_count", new_count)
        return new_count

    async def get_user_id(self, agentfs) -> Optional[str]:
        """Obtém user_id da sessão (None = guest)."""
        try:
            return await agentfs.kv.get("session:user_id")
        except Exception:
            return None

    async def set_user_id(self, agentfs, user_id: str) -> None:
        """Associa user_id à sessão (após signup/login)."""
        await agentfs.kv.set("session:user_id", user_id)
        logger.info("User ID set for session", user_id=user_id)

    async def check_limit(self, agentfs) -> GuestCheckResult:
        """Verifica limite SEM incrementar contador.

        Use para verificar antes de mostrar UI.
        """
        # Se tem user_id, sempre permitir
        user_id = await self.get_user_id(agentfs)
        if user_id:
            return GuestCheckResult(
                action=GuestLimitAction.ALLOW,
                prompt_count=await self.get_prompt_count(agentfs),
                prompts_remaining=-1,  # Ilimitado para usuários autenticados
                can_continue=True,
            )

        # Se ilimitado, sempre permitir
        if self.is_unlimited():
            return GuestCheckResult(
                action=GuestLimitAction.ALLOW,
                prompt_count=await self.get_prompt_count(agentfs),
                prompts_remaining=-1,
                message="Modo desenvolvimento - sem limites",
                can_continue=True,
            )

        prompt_count = await self.get_prompt_count(agentfs)
        limit = self.config.prompts_limit
        grace = self.config.grace_prompts
        total_allowed = limit + grace

        prompts_remaining = max(0, total_allowed - prompt_count)

        # Verificar limites
        if prompt_count >= total_allowed:
            return GuestCheckResult(
                action=GuestLimitAction.REQUIRE_SIGNUP,
                prompt_count=prompt_count,
                prompts_remaining=0,
                message="Crie uma conta para continuar a conversa",
                can_continue=False,
            )

        if prompt_count >= limit:
            # Dentro do grace period
            return GuestCheckResult(
                action=GuestLimitAction.SOFT_LIMIT,
                prompt_count=prompt_count,
                prompts_remaining=prompts_remaining,
                message=f"Você tem mais {prompts_remaining} mensagem(ns) antes de criar conta",
                can_continue=True,
            )

        # Verificar soft limit (aviso prévio)
        soft = self.config.soft_limit_threshold
        if soft > 0 and prompt_count >= soft:
            return GuestCheckResult(
                action=GuestLimitAction.SOFT_LIMIT,
                prompt_count=prompt_count,
                prompts_remaining=prompts_remaining,
                message=f"Restam {prompts_remaining} mensagem(ns) gratuita(s)",
                can_continue=True,
            )

        return GuestCheckResult(
            action=GuestLimitAction.ALLOW,
            prompt_count=prompt_count,
            prompts_remaining=prompts_remaining,
            can_continue=True,
        )

    async def check_and_increment(self, agentfs) -> GuestCheckResult:
        """Verifica limite E incrementa contador.

        Use ao processar um prompt.
        Fluxo:
        1. Verifica limite ANTES de incrementar
        2. Se permitido, incrementa contador
        3. Retorna resultado com novo count
        """
        # Verificar primeiro
        result = await self.check_limit(agentfs)

        if not result.can_continue:
            return result

        # Incrementar contador
        new_count = await self.increment_prompt_count(agentfs)

        # Recalcular resultado com novo count
        if self.is_unlimited():
            return GuestCheckResult(
                action=GuestLimitAction.ALLOW,
                prompt_count=new_count,
                prompts_remaining=-1,
                can_continue=True,
            )

        limit = self.config.prompts_limit
        grace = self.config.grace_prompts
        total_allowed = limit + grace
        prompts_remaining = max(0, total_allowed - new_count)

        # Verificar se próximo prompt vai exigir signup
        if new_count >= total_allowed:
            return GuestCheckResult(
                action=GuestLimitAction.REQUIRE_SIGNUP,
                prompt_count=new_count,
                prompts_remaining=0,
                message="Crie uma conta para continuar a conversa",
                can_continue=False,  # Não pode mais
            )

        if new_count >= limit:
            return GuestCheckResult(
                action=GuestLimitAction.SOFT_LIMIT,
                prompt_count=new_count,
                prompts_remaining=prompts_remaining,
                message=f"Você tem mais {prompts_remaining} mensagem(ns) antes de criar conta",
                can_continue=True,
            )

        return GuestCheckResult(
            action=GuestLimitAction.ALLOW,
            prompt_count=new_count,
            prompts_remaining=prompts_remaining,
            can_continue=True,
        )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_guest_limit_manager: Optional[GuestLimitManager] = None


def get_guest_limit_manager() -> GuestLimitManager:
    """Obtém instância global do GuestLimitManager."""
    global _guest_limit_manager
    if _guest_limit_manager is None:
        _guest_limit_manager = GuestLimitManager()
    return _guest_limit_manager


def reset_guest_limit_manager(config: Optional[GuestLimitConfig] = None) -> GuestLimitManager:
    """Reseta o manager com nova configuração (útil para testes)."""
    global _guest_limit_manager
    _guest_limit_manager = GuestLimitManager(config)
    return _guest_limit_manager
