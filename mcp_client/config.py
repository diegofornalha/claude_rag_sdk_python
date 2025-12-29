"""
MCP Configuration - Configuração central para adapters MCP

Este arquivo define quais adapters estão habilitados e suas configurações.
Para desativar um adapter, basta definir enabled=False.
Para remover completamente, basta não registrar o adapter.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPAdapterConfig:
    """Configuração de um adapter MCP específico"""

    enabled: bool = True
    command: list[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPConfig:
    """Configuração global do sistema MCP"""

    # Adapters registrados e suas configurações
    adapters: dict[str, MCPAdapterConfig] = field(default_factory=dict)

    # Configurações globais
    default_timeout: float = 30.0
    max_concurrent_connections: int = 5
    log_level: str = "INFO"

    # Ingestão
    ingest_delay_between_calls: float = 0.5  # Rate limiting
    ingest_batch_size: int = 10

    @classmethod
    def default(cls) -> "MCPConfig":
        """
        Retorna configuração padrão com adapters conhecidos.

        Por padrão, adapters são DESABILITADOS.
        O usuário deve habilitar explicitamente os que deseja usar.
        """
        return cls(
            adapters={
                "angular-cli": MCPAdapterConfig(
                    enabled=False,  # Desabilitado por padrão
                    command=["npx", "-y", "@angular/cli", "mcp"],
                    timeout_seconds=60.0,
                    options={
                        "read_only": False,
                        "local_only": False,
                    },
                ),
                # Futuros adapters podem ser adicionados aqui
                # "react-devtools": MCPAdapterConfig(enabled=False, ...),
                # "vue-cli": MCPAdapterConfig(enabled=False, ...),
            }
        )

    @classmethod
    def from_env(cls) -> "MCPConfig":
        """
        Carrega configuração de variáveis de ambiente.

        Variáveis:
        - MCP_ANGULAR_CLI_ENABLED: "true" ou "false"
        - MCP_CONFIG_PATH: Caminho para arquivo JSON de config
        """
        config = cls.default()

        # Carrega de arquivo se especificado
        config_path = os.getenv("MCP_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            config = cls.from_file(config_path)

        # Override individual por env var
        if os.getenv("MCP_ANGULAR_CLI_ENABLED", "").lower() == "true":
            if "angular-cli" in config.adapters:
                config.adapters["angular-cli"].enabled = True

        return config

    @classmethod
    def from_file(cls, path: str) -> "MCPConfig":
        """Carrega configuração de arquivo JSON."""
        with open(path) as f:
            data = json.load(f)

        config = cls.default()

        # Atualiza adapters
        for name, adapter_data in data.get("adapters", {}).items():
            if name in config.adapters:
                config.adapters[name] = MCPAdapterConfig(
                    enabled=adapter_data.get("enabled", False),
                    command=adapter_data.get("command", config.adapters[name].command),
                    timeout_seconds=adapter_data.get("timeout_seconds", 30.0),
                    options=adapter_data.get("options", {}),
                )
            else:
                # Adapter customizado
                config.adapters[name] = MCPAdapterConfig(
                    enabled=adapter_data.get("enabled", False),
                    command=adapter_data.get("command", []),
                    timeout_seconds=adapter_data.get("timeout_seconds", 30.0),
                    options=adapter_data.get("options", {}),
                )

        # Atualiza configs globais
        config.default_timeout = data.get("default_timeout", config.default_timeout)
        config.max_concurrent_connections = data.get(
            "max_concurrent_connections", config.max_concurrent_connections
        )
        config.ingest_delay_between_calls = data.get(
            "ingest_delay_between_calls", config.ingest_delay_between_calls
        )

        return config

    def to_file(self, path: str) -> None:
        """Salva configuração em arquivo JSON."""
        data = {
            "adapters": {
                name: {
                    "enabled": adapter.enabled,
                    "command": adapter.command,
                    "timeout_seconds": adapter.timeout_seconds,
                    "options": adapter.options,
                }
                for name, adapter in self.adapters.items()
            },
            "default_timeout": self.default_timeout,
            "max_concurrent_connections": self.max_concurrent_connections,
            "ingest_delay_between_calls": self.ingest_delay_between_calls,
            "ingest_batch_size": self.ingest_batch_size,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def is_adapter_enabled(self, name: str) -> bool:
        """Verifica se um adapter está habilitado."""
        adapter = self.adapters.get(name)
        return adapter is not None and adapter.enabled

    def get_enabled_adapters(self) -> list[str]:
        """Retorna lista de adapters habilitados."""
        return [name for name, config in self.adapters.items() if config.enabled]

    def enable_adapter(self, name: str) -> bool:
        """Habilita um adapter."""
        if name in self.adapters:
            self.adapters[name].enabled = True
            return True
        return False

    def disable_adapter(self, name: str) -> bool:
        """Desabilita um adapter."""
        if name in self.adapters:
            self.adapters[name].enabled = False
            return True
        return False


# Configuração global singleton (carregada uma vez)
_global_config: MCPConfig | None = None


def get_mcp_config() -> MCPConfig:
    """Obtém a configuração global do MCP."""
    global _global_config
    if _global_config is None:
        _global_config = MCPConfig.from_env()
    return _global_config


def reload_mcp_config() -> MCPConfig:
    """Recarrega a configuração do MCP."""
    global _global_config
    _global_config = MCPConfig.from_env()
    return _global_config
