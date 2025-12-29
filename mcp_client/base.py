"""
MCP Base Classes - Interfaces e tipos base para o sistema MCP

Este módulo define as interfaces que todos os adapters MCP devem implementar.
O core MCP não depende de nenhum adapter específico.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPAdapterStatus(Enum):
    """Status de um adapter MCP"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPToolResult:
    """Resultado de uma chamada de tool MCP"""

    success: bool
    data: Any = None
    error: str | None = None
    tool_name: str | None = None
    execution_time_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool_name": self.tool_name,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class MCPDocument:
    """Documento extraído de um MCP server para ingestão"""

    content: str
    title: str
    source: str
    doc_type: str = "markdown"
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPAdapterInfo:
    """Informações sobre um adapter MCP"""

    name: str
    description: str
    version: str
    tools: list[str]
    enabled: bool = True
    status: MCPAdapterStatus = MCPAdapterStatus.DISCONNECTED


class BaseMCPAdapter(ABC):
    """
    Interface base para todos os adapters MCP.

    Cada MCP server (Angular CLI, outros futuros) deve implementar
    esta interface para ser registrado no sistema.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome único do adapter (ex: 'angular-cli')"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Descrição do adapter"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Versão do adapter"""
        pass

    @property
    @abstractmethod
    def command(self) -> list[str]:
        """Comando para iniciar o MCP server"""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """Conecta ao MCP server"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Desconecta do MCP server"""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Verifica se está conectado"""
        pass

    @abstractmethod
    async def list_tools(self) -> list[str]:
        """Lista tools disponíveis no MCP server"""
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Chama uma tool do MCP server"""
        pass

    @abstractmethod
    async def get_documents_for_ingest(self, **kwargs) -> list[MCPDocument]:
        """
        Obtém documentos do MCP para ingestão no RAG.

        Cada adapter implementa sua lógica específica de extração.
        Por exemplo, Angular CLI pode buscar docs, examples, best practices.
        """
        pass

    def get_info(self) -> MCPAdapterInfo:
        """Retorna informações do adapter"""
        return MCPAdapterInfo(
            name=self.name,
            description=self.description,
            version=self.version,
            tools=[],  # Preenchido após conexão
            enabled=True,
            status=MCPAdapterStatus.DISCONNECTED,
        )


class MCPIngestCapability(ABC):
    """
    Mixin para adapters que suportam ingestão de documentos.

    Define métodos específicos para extração de conteúdo
    que será ingerido no RAG.
    """

    @abstractmethod
    async def search_documentation(self, query: str) -> list[MCPDocument]:
        """Busca documentação"""
        pass

    @abstractmethod
    async def get_examples(self, topic: str) -> list[MCPDocument]:
        """Obtém exemplos de código"""
        pass

    @abstractmethod
    async def get_best_practices(self) -> list[MCPDocument]:
        """Obtém guia de best practices"""
        pass
