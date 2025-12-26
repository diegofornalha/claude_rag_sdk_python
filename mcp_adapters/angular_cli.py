"""
Angular CLI MCP Adapter - Plugin para integração com Angular CLI MCP Server

Este arquivo é um PLUGIN OPCIONAL. Pode ser removido sem afetar o sistema.

Para remover:
1. Delete este arquivo
2. Remova a referência em __init__.py
3. Remova "angular-cli" da configuração
O sistema continuará funcionando normalmente.

Tools disponíveis do Angular CLI MCP:
- search_documentation: Busca na documentação oficial do Angular
- find_examples: Encontra exemplos de código curados
- get_beices: Retorna o Angular Best Practices Guide
- list_projects: Lista projetos no workspace Angular
- ai_tutor: Tutor interativo de Angular
- onpush_zoneless_migration: Análise para migração zoneless
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from claude_rag_sdk.mcp import (
    BaseMCPAdapter,
    MCPAdapterConfig,
    MCPAdapterInfo,
    MCPAdapterStatus,
    MCPClient,
    MCPClientConfig,
    MCPDocument,
    MCPIngestCapability,
    MCPToolResult,
    register_adapter,
)

logger = logging.getLogger(__name__)


# Queries padrão para ingestão de documentação Angular
DEFAULT_DOC_QUERIES = [
    "standalone components",
    "signals",
    "control flow @if @for @switch",
    "dependency injection providers",
    "routing guards",
    "lazy loading routes",
    "reactive forms",
    "form validation",
    "http client interceptors",
    "change detection onpush",
    "zoneless applications",
    "defer loading",
    "testing components",
    "ng generate schematics",
]


class AngularCLIMCPAdapter(BaseMCPAdapter, MCPIngestCapability):
    """
    Adapter para Angular CLI MCP Server.

    Permite:
    - Buscar documentação oficial do Angular
    - Obter exemplos de código
    - Acessar best practices
    - Listar projetos do workspace

    Configuração:
        enabled: true/false
        command: ["npx", "-y", "@angular/cli", "mcp"]
        options:
            read_only: false
            local_only: false
    """

    def __init__(self, config: Optional[MCPAdapterConfig] = None):
        self._config = config or MCPAdapterConfig(
            enabled=False,
            command=["npx", "-y", "@angular/cli", "mcp"],
        )
        self._client: Optional[MCPClient] = None
        self._available_tools: List[str] = []

    @property
    def name(self) -> str:
        return "angular-cli"

    @property
    def description(self) -> str:
        return "Angular CLI MCP Server - Documentação, exemplos e best practices do Angular"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def command(self) -> List[str]:
        cmd = self._config.command.copy()
        # Adiciona opções se configuradas
        if self._config.options.get("read_only"):
            cmd.append("--read-only")
        if self._config.options.get("local_only"):
            cmd.append("--local-only")
        return cmd

    async def connect(self) -> bool:
        """Conecta ao Angular CLI MCP Server."""
        if self._client and await self._client.is_connected():
            return True

        client_config = MCPClientConfig(
            command=self.command,
            timeout_seconds=self._config.timeout_seconds,
        )

        self._client = MCPClient(client_config)
        connected = await self._client.connect()

        if connected:
            self._available_tools = self._client.available_tools
            logger.info(f"Angular CLI MCP connected. Tools: {self._available_tools}")

        return connected

    async def disconnect(self) -> None:
        """Desconecta do Angular CLI MCP Server."""
        if self._client:
            await self._client.disconnect()
            self._client = None
            self._available_tools = []

    async def is_connected(self) -> bool:
        """Verifica se está conectado."""
        return self._client is not None and await self._client.is_connected()

    async def list_tools(self) -> List[str]:
        """Lista tools disponíveis."""
        return self._available_tools.copy()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Chama uma tool do Angular CLI MCP."""
        if not self._client:
            return MCPToolResult(
                success=False,
                error="Not connected",
                tool_name=tool_name,
            )
        return await self._client.call_tool(tool_name, arguments)

    def get_info(self) -> MCPAdapterInfo:
        """Retorna informações do adapter."""
        status = MCPAdapterStatus.DISCONNECTED
        if self._client:
            status = self._client.status

        return MCPAdapterInfo(
            name=self.name,
            description=self.description,
            version=self.version,
            tools=self._available_tools,
            enabled=self._config.enabled,
            status=status,
        )

    # === MCPIngestCapability Implementation ===

    async def search_documentation(self, query: str) -> List[MCPDocument]:
        """
        Busca na documentação oficial do Angular.

        Args:
            query: Termo de busca (ex: "signals", "standalone components")

        Returns:
            Lista de MCPDocument com a documentação encontrada
        """
        result = await self.call_tool("search_documentation", {"query": query})

        if not result.success or not result.data:
            logger.warning(f"No results for query: {query}")
            return []

        # Processa resultado
        content = result.data if isinstance(result.data, str) else str(result.data)

        return [
            MCPDocument(
                content=content,
                title=f"Angular Docs: {query}",
                source=f"angular-cli-mcp:search_documentation:{query}",
                doc_type="markdown",
                url="https://angular.dev",
                metadata={
                    "query": query,
                    "tool": "search_documentation",
                    "mcp_server": self.name,
                    "fetched_at": datetime.now().isoformat(),
                },
            )
        ]

    async def get_examples(self, topic: str) -> List[MCPDocument]:
        """
        Obtém exemplos de código para um tópico.

        Args:
            topic: Tópico (ex: "forms", "http", "routing")

        Returns:
            Lista de MCPDocument com exemplos de código
        """
        result = await self.call_tool("find_examples", {"topic": topic})

        if not result.success or not result.data:
            logger.warning(f"No examples for topic: {topic}")
            return []

        content = result.data if isinstance(result.data, str) else str(result.data)

        return [
            MCPDocument(
                content=content,
                title=f"Angular Examples: {topic}",
                source=f"angular-cli-mcp:find_examples:{topic}",
                doc_type="markdown",
                metadata={
                    "topic": topic,
                    "tool": "find_examples",
                    "mcp_server": self.name,
                    "fetched_at": datetime.now().isoformat(),
                },
            )
        ]

    async def get_best_practices(self) -> List[MCPDocument]:
        """
        Obtém o Angular Best Practices Guide.

        Returns:
            Lista com documento de best practices
        """
        result = await self.call_tool("get_beices", {})

        if not result.success or not result.data:
            logger.warning("Failed to get best practices")
            return []

        content = result.data if isinstance(result.data, str) else str(result.data)

        return [
            MCPDocument(
                content=content,
                title="Angular Best Practices Guide",
                source="angular-cli-mcp:get_beices",
                doc_type="markdown",
                metadata={
                    "tool": "get_beices",
                    "mcp_server": self.name,
                    "fetched_at": datetime.now().isoformat(),
                },
            )
        ]

    async def get_documents_for_ingest(
        self,
        queries: Optional[List[str]] = None,
        include_examples: bool = True,
        include_best_practices: bool = True,
    ) -> List[MCPDocument]:
        """
        Obtém todos os documentos para ingestão no RAG.

        Args:
            queries: Lista de queries para buscar docs (usa padrão se None)
            include_examples: Se deve incluir exemplos de código
            include_best_practices: Se deve incluir best practices

        Returns:
            Lista de todos os documentos coletados
        """
        documents: List[MCPDocument] = []

        # Queries para documentação
        search_queries = queries or DEFAULT_DOC_QUERIES

        # Busca documentação
        for query in search_queries:
            try:
                docs = await self.search_documentation(query)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error searching docs for '{query}': {e}")

        # Busca exemplos
        if include_examples:
            example_topics = [
                "standalone components",
                "signals",
                "forms",
                "http",
                "routing",
            ]
            for topic in example_topics:
                try:
                    examples = await self.get_examples(topic)
                    documents.extend(examples)
                except Exception as e:
                    logger.error(f"Error getting examples for '{topic}': {e}")

        # Best practices
        if include_best_practices:
            try:
                practices = await self.get_best_practices()
                documents.extend(practices)
            except Exception as e:
                logger.error(f"Error getting best practices: {e}")

        logger.info(f"Collected {len(documents)} documents from Angular CLI MCP")
        return documents

    # === Métodos específicos do Angular CLI ===

    async def list_projects(self) -> MCPToolResult:
        """Lista projetos no workspace Angular."""
        return await self.call_tool("list_projects", {})

    async def get_migration_plan(self, code: str) -> MCPToolResult:
        """
        Obtém plano de migração para OnPush/Zoneless.

        Args:
            code: Código Angular a ser analisado
        """
        return await self.call_tool("onpush_zoneless_migration", {"code": code})


# === Auto-registro do adapter ===
# Quando este arquivo é importado, o adapter é automaticamente
# registrado no registry global

try:
    register_adapter("angular-cli", AngularCLIMCPAdapter)
    logger.debug("Angular CLI adapter registered")
except Exception as e:
    logger.error(f"Failed to register Angular CLI adapter: {e}")
