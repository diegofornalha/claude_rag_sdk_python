"""
MCP Module - Sistema de integração com MCP Servers

Este módulo fornece o CORE para comunicação com MCP servers.
Não depende de nenhum adapter específico.

Uso básico:
    from claude_rag_sdk.mcp_client import get_mcp_registry, get_adapter

    # Lista adapters disponíveis
    registry = get_mcp_registry()
    print(registry.list_enabled())

    # Obtém adapter (se habilitado)
    adapter = await get_adapter("angular-cli")
    result = await adapter.call_tool("search_documentation", {"query": "signals"})

Configuração:
    - Via env var: MCP_ANGULAR_CLI_ENABLED=true
    - Via arquivo: MCP_CONFIG_PATH=/path/to/mcp_config.json
    - Programaticamente: config.enable_adapter("angular-cli")
"""

from .base import (
    BaseMCPAdapter,
    MCPAdapterInfo,
    MCPAdapterStatus,
    MCPDocument,
    MCPIngestCapability,
    MCPToolResult,
)
from .client import (
    MCPClient,
    MCPClientConfig,
)
from .config import (
    MCPAdapterConfig,
    MCPConfig,
    get_mcp_config,
    reload_mcp_config,
)
from .registry import (
    MCPAdapterDisabledError,
    MCPAdapterNotFoundError,
    MCPRegistry,
    get_adapter,
    get_mcp_registry,
    register_adapter,
)

__all__ = [
    # Base
    "BaseMCPAdapter",
    "MCPAdapterInfo",
    "MCPAdapterStatus",
    "MCPDocument",
    "MCPIngestCapability",
    "MCPToolResult",
    # Client
    "MCPClient",
    "MCPClientConfig",
    # Config
    "MCPAdapterConfig",
    "MCPConfig",
    "get_mcp_config",
    "reload_mcp_config",
    # Registry
    "MCPAdapterDisabledError",
    "MCPAdapterNotFoundError",
    "MCPRegistry",
    "get_adapter",
    "get_mcp_registry",
    "register_adapter",
]
