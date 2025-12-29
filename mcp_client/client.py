"""
MCP Client - Cliente genérico para comunicação com MCP servers

Este é o CORE do sistema MCP. Não depende de nenhum adapter específico.
Implementa o protocolo JSON-RPC 2.0 sobre stdio.
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from .base import MCPAdapterStatus, MCPToolResult

logger = logging.getLogger(__name__)


@dataclass
class MCPClientConfig:
    """Configuração do cliente MCP"""

    command: list[str]
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class MCPClient:
    """
    Cliente MCP genérico que implementa o protocolo JSON-RPC 2.0.

    Este cliente pode se conectar a qualquer MCP server que siga
    o protocolo padrão. Não tem conhecimento de servers específicos.
    """

    def __init__(self, config: MCPClientConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._status = MCPAdapterStatus.DISCONNECTED
        self._available_tools: list[str] = []
        self._lock = asyncio.Lock()

    @property
    def status(self) -> MCPAdapterStatus:
        return self._status

    @property
    def available_tools(self) -> list[str]:
        return self._available_tools.copy()

    async def connect(self) -> bool:
        """
        Inicia conexão com o MCP server.

        Returns:
            True se conectou com sucesso, False caso contrário.
        """
        async with self._lock:
            if self._status == MCPAdapterStatus.CONNECTED:
                return True

            self._status = MCPAdapterStatus.CONNECTING

            try:
                # Inicia o processo MCP
                self._process = subprocess.Popen(
                    self.config.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Aguarda um momento para o processo iniciar
                await asyncio.sleep(0.5)

                # Verifica se o processo ainda está rodando
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read() if self._process.stderr else ""
                    logger.error(f"MCP process terminated: {stderr}")
                    self._status = MCPAdapterStatus.ERROR
                    return False

                # Envia mensagem de inicialização
                init_result = await self._initialize()
                if not init_result:
                    self._status = MCPAdapterStatus.ERROR
                    return False

                # Lista tools disponíveis
                await self._fetch_tools()

                self._status = MCPAdapterStatus.CONNECTED
                logger.info(f"MCP connected with {len(self._available_tools)} tools")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to MCP: {e}")
                self._status = MCPAdapterStatus.ERROR
                await self.disconnect()
                return False

    async def disconnect(self) -> None:
        """Encerra conexão com o MCP server."""
        async with self._lock:
            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error terminating MCP process: {e}")
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                finally:
                    self._process = None

            self._status = MCPAdapterStatus.DISCONNECTED
            self._available_tools = []

    async def is_connected(self) -> bool:
        """Verifica se está conectado ao MCP server."""
        if self._status != MCPAdapterStatus.CONNECTED:
            return False
        if self._process is None or self._process.poll() is not None:
            self._status = MCPAdapterStatus.DISCONNECTED
            return False
        return True

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """
        Chama uma tool do MCP server.

        Args:
            tool_name: Nome da tool a ser chamada
            arguments: Argumentos para a tool

        Returns:
            MCPToolResult com o resultado da chamada
        """
        if not await self.is_connected():
            return MCPToolResult(
                success=False,
                error="Not connected to MCP server",
                tool_name=tool_name,
            )

        start_time = time.time()

        try:
            response = await self._send_request(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments,
                },
            )

            execution_time = (time.time() - start_time) * 1000

            if "error" in response:
                return MCPToolResult(
                    success=False,
                    error=response["error"].get("message", "Unknown error"),
                    tool_name=tool_name,
                    execution_time_ms=execution_time,
                )

            result = response.get("result", {})
            content = result.get("content", [])

            # Extrai texto do conteúdo
            data = self._extract_content(content)

            return MCPToolResult(
                success=True,
                data=data,
                tool_name=tool_name,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Error calling tool {tool_name}: {e}")
            return MCPToolResult(
                success=False,
                error=str(e),
                tool_name=tool_name,
                execution_time_ms=execution_time,
            )

    async def _initialize(self) -> bool:
        """Envia mensagem de inicialização para o MCP server."""
        try:
            response = await self._send_request(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "claude-rag-mcp-client",
                        "version": "1.0.0",
                    },
                },
            )

            if "error" in response:
                logger.error(f"Initialize failed: {response['error']}")
                return False

            # Envia notificação initialized
            await self._send_notification("notifications/initialized")
            return True

        except Exception as e:
            logger.error(f"Initialize error: {e}")
            return False

    async def _fetch_tools(self) -> None:
        """Obtém lista de tools disponíveis do MCP server."""
        try:
            response = await self._send_request(method="tools/list", params={})

            if "result" in response:
                tools = response["result"].get("tools", [])
                self._available_tools = [t.get("name", "") for t in tools]
                logger.info(f"Available tools: {self._available_tools}")

        except Exception as e:
            logger.error(f"Failed to fetch tools: {e}")

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Envia uma requisição JSON-RPC e aguarda resposta."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP process not running")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        request_str = json.dumps(request) + "\n"

        # Envia requisição
        self._process.stdin.write(request_str)
        self._process.stdin.flush()

        # Lê resposta com timeout
        response_line = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, self._process.stdout.readline),
            timeout=self.config.timeout_seconds,
        )

        if not response_line:
            raise RuntimeError("Empty response from MCP server")

        return json.loads(response_line)

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Envia uma notificação (sem esperar resposta)."""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        notification_str = json.dumps(notification) + "\n"
        self._process.stdin.write(notification_str)
        self._process.stdin.flush()

    def _extract_content(self, content: list[dict]) -> Any:
        """Extrai dados úteis do conteúdo da resposta MCP."""
        if not content:
            return None

        # Se há apenas um item de texto, retorna o texto
        if len(content) == 1 and content[0].get("type") == "text":
            return content[0].get("text", "")

        # Caso contrário, retorna lista de textos
        texts = []
        for item in content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))

        return texts if len(texts) > 1 else (texts[0] if texts else None)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
