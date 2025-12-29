"""
MCP Registry - Sistema de registro e gerenciamento de adapters MCP

O registry é responsável por:
- Registrar adapters disponíveis
- Carregar apenas adapters habilitados
- Gerenciar ciclo de vida (connect/disconnect)
- Prover acesso aos adapters de forma segura
"""

import asyncio
import logging

from .base import BaseMCPAdapter, MCPAdapterInfo, MCPAdapterStatus
from .config import MCPConfig, get_mcp_config

logger = logging.getLogger(__name__)


class MCPAdapterNotFoundError(Exception):
    """Adapter não encontrado no registry."""

    pass


class MCPAdapterDisabledError(Exception):
    """Adapter está desabilitado na configuração."""

    pass


class MCPRegistry:
    """
    Registry central para adapters MCP.

    Permite registrar, descobrir e gerenciar adapters de forma
    desacoplada. Novos adapters podem ser adicionados sem modificar
    o código existente.
    """

    def __init__(self, config: MCPConfig | None = None):
        self._config = config or get_mcp_config()
        self._adapter_classes: dict[str, type[BaseMCPAdapter]] = {}
        self._active_adapters: dict[str, BaseMCPAdapter] = {}
        self._lock = asyncio.Lock()

    @property
    def config(self) -> MCPConfig:
        return self._config

    def register_adapter(self, name: str, adapter_class: type[BaseMCPAdapter]) -> None:
        """
        Registra uma classe de adapter no registry.

        Args:
            name: Nome único do adapter (ex: 'angular-cli')
            adapter_class: Classe que implementa BaseMCPAdapter

        Note:
            Registrar não significa habilitar. O adapter só será
            instanciado se estiver enabled na configuração.
        """
        if not issubclass(adapter_class, BaseMCPAdapter):
            raise TypeError(f"{adapter_class} must inherit from BaseMCPAdapter")

        self._adapter_classes[name] = adapter_class
        logger.debug(f"Registered adapter: {name}")

    def unregister_adapter(self, name: str) -> bool:
        """
        Remove um adapter do registry.

        Se o adapter estiver ativo, será desconectado primeiro.
        """
        if name in self._adapter_classes:
            # Se estiver ativo, marca para remoção (será desconectado depois)
            del self._adapter_classes[name]
            logger.debug(f"Unregistered adapter: {name}")
            return True
        return False

    def list_registered(self) -> list[str]:
        """Lista todos os adapters registrados (habilitados ou não)."""
        return list(self._adapter_classes.keys())

    def list_enabled(self) -> list[str]:
        """Lista apenas adapters habilitados na configuração."""
        return [name for name in self._adapter_classes if self._config.is_adapter_enabled(name)]

    def list_active(self) -> list[str]:
        """Lista adapters atualmente conectados."""
        return list(self._active_adapters.keys())

    def is_registered(self, name: str) -> bool:
        """Verifica se um adapter está registrado."""
        return name in self._adapter_classes

    def is_enabled(self, name: str) -> bool:
        """Verifica se um adapter está habilitado."""
        return name in self._adapter_classes and self._config.is_adapter_enabled(name)

    def is_active(self, name: str) -> bool:
        """Verifica se um adapter está ativo (conectado)."""
        return name in self._active_adapters

    async def get_adapter(self, name: str) -> BaseMCPAdapter:
        """
        Obtém uma instância ativa de um adapter.

        Se o adapter não estiver ativo, tenta conectar.

        Args:
            name: Nome do adapter

        Returns:
            Instância do adapter conectado

        Raises:
            MCPAdapterNotFoundError: Se adapter não está registrado
            MCPAdapterDisabledError: Se adapter está desabilitado
        """
        if not self.is_registered(name):
            raise MCPAdapterNotFoundError(f"Adapter '{name}' not registered")

        if not self.is_enabled(name):
            raise MCPAdapterDisabledError(
                f"Adapter '{name}' is disabled. "
                f"Enable it in config or set MCP_{name.upper().replace('-', '_')}_ENABLED=true"
            )

        async with self._lock:
            # Se já está ativo, retorna
            if name in self._active_adapters:
                adapter = self._active_adapters[name]
                if await adapter.is_connected():
                    return adapter
                # Se desconectou, remove e recria
                del self._active_adapters[name]

            # Cria nova instância e conecta
            adapter_class = self._adapter_classes[name]
            adapter_config = self._config.adapters.get(name)

            adapter = adapter_class(adapter_config)
            connected = await adapter.connect()

            if not connected:
                raise RuntimeError(f"Failed to connect to adapter '{name}'")

            self._active_adapters[name] = adapter
            return adapter

    async def disconnect_adapter(self, name: str) -> bool:
        """Desconecta um adapter específico."""
        async with self._lock:
            if name in self._active_adapters:
                adapter = self._active_adapters[name]
                await adapter.disconnect()
                del self._active_adapters[name]
                logger.info(f"Disconnected adapter: {name}")
                return True
        return False

    async def disconnect_all(self) -> None:
        """Desconecta todos os adapters ativos."""
        async with self._lock:
            for name in list(self._active_adapters.keys()):
                adapter = self._active_adapters[name]
                try:
                    await adapter.disconnect()
                except Exception as e:
                    logger.error(f"Error disconnecting {name}: {e}")
                del self._active_adapters[name]
            logger.info("All adapters disconnected")

    def get_adapter_info(self, name: str) -> MCPAdapterInfo | None:
        """Obtém informações de um adapter."""
        if name not in self._adapter_classes:
            return None

        adapter_class = self._adapter_classes[name]
        config = self._config.adapters.get(name)

        # Cria instância temporária só para obter info
        temp_adapter = adapter_class(config)
        info = temp_adapter.get_info()
        info.enabled = self._config.is_adapter_enabled(name)

        # Se está ativo, atualiza status
        if name in self._active_adapters:
            info.status = MCPAdapterStatus.CONNECTED
            info.tools = self._active_adapters[name]._available_tools

        return info

    def get_all_adapters_info(self) -> list[MCPAdapterInfo]:
        """Obtém informações de todos os adapters registrados."""
        return [
            self.get_adapter_info(name)
            for name in self._adapter_classes
            if self.get_adapter_info(name) is not None
        ]


# Registry global singleton
_global_registry: MCPRegistry | None = None


def get_mcp_registry() -> MCPRegistry:
    """Obtém o registry global de adapters MCP."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MCPRegistry()
    return _global_registry


def register_adapter(name: str, adapter_class: type[BaseMCPAdapter]) -> None:
    """Atalho para registrar adapter no registry global."""
    get_mcp_registry().register_adapter(name, adapter_class)


async def get_adapter(name: str) -> BaseMCPAdapter:
    """Atalho para obter adapter do registry global."""
    return await get_mcp_registry().get_adapter(name)
