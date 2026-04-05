"""
mcp_client.py
-------------
Cliente MCP que lanza mcp_server.py como subproceso via stdio.
Usa un thread dedicado para el event loop async para evitar conflictos
de cancel scope entre asyncio y anyio (Python 3.10).

Uso:
    from mcp_client import MCPPipeline

    with MCPPipeline() as p:
        p.fetch_gdelt("2023-01-01", "2023-12-31")
        p.fetch_rss("bce", "2023-01-01", "2023-12-31")

Requiere:
    pip install "mcp[cli]"
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"
PYTHON_EXE = sys.executable


class MCPPipeline:
    """
    Wrapper sincrono sobre el cliente MCP async.
    El event loop corre en un thread separado (pattern run_coroutine_threadsafe)
    para evitar conflictos de cancel scope de anyio en Python 3.10.
    """

    def __init__(self, timeout: int = 120):
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._session: ClientSession | None = None
        self._exit_stack = None

        # Conectar sincrono
        future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        future.result(timeout=30)

    async def _connect(self):
        from contextlib import AsyncExitStack

        server_params = StdioServerParameters(
            command=PYTHON_EXE,
            args=[str(SERVER_SCRIPT)],
        )
        self._exit_stack = AsyncExitStack()
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        tools = await self._session.list_tools()
        tool_names = [t.name for t in tools.tools]
        print(f"[MCP] Conectado. Tools: {tool_names}")

    async def _call_tool(self, name: str, arguments: dict) -> str:
        if self._session is None:
            raise RuntimeError("Session not initialized")
        result = await self._session.call_tool(name, arguments)
        return result.content[0].text

    def _run(self, coro):
        """Ejecuta una corrutina en el thread del event loop y espera resultado."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self._timeout)

    # ── Metodos publicos ──────────────────────────────────────

    def fetch_gdelt(self, start_date: str, end_date: str) -> list[dict]:
        """Descarga GDELT v2 cuantitativo mensual."""
        raw = self._run(self._call_tool("fetch_gdelt_spain", {
            "start_date": start_date,
            "end_date": end_date,
        }))
        return json.loads(raw)

    def fetch_rss(self, source: str, start_date: str, end_date: str) -> dict:
        """Descarga RSS de fuente oficial (bce|ine|bde)."""
        raw = self._run(self._call_tool("fetch_rss_official", {
            "source": source,
            "start_date": start_date,
            "end_date": end_date,
        }))
        return json.loads(raw)

    def search(self, query: str, start_date: str, end_date: str) -> list[dict]:
        """Busca en todas las fuentes almacenadas."""
        raw = self._run(self._call_tool("search_news", {
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
        }))
        return json.loads(raw)

    def get_macro(
        self, topic: str, country: str, start_date: str, end_date: str
    ) -> list[dict]:
        """Noticias macro por tema."""
        raw = self._run(self._call_tool("get_macro_news", {
            "topic": topic,
            "country": country,
            "start_date": start_date,
            "end_date": end_date,
        }))
        return json.loads(raw)

    def get_entity(self, entity: str, start_date: str, end_date: str) -> list[dict]:
        """Noticias por entidad (bce|ine|bde)."""
        raw = self._run(self._call_tool("get_entity_news", {
            "entity": entity,
            "start_date": start_date,
            "end_date": end_date,
        }))
        return json.loads(raw)

    # ── Lifecycle ─────────────────────────────────────────────

    def close(self):
        """Cierra la sesion MCP y el event loop."""
        if self._exit_stack:
            future = asyncio.run_coroutine_threadsafe(
                self._exit_stack.aclose(), self._loop
            )
            try:
                future.result(timeout=10)
            except Exception:
                pass  # El servidor ya proceso todo, ignorar error de cleanup

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── CLI test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Test MCP Client ===")
    with MCPPipeline() as pipeline:
        print("Cliente OK. Metodos disponibles:")
        print("  pipeline.fetch_gdelt(start, end)")
        print("  pipeline.fetch_rss(source, start, end)")
        print("  pipeline.search(query, start, end)")
        print("  pipeline.get_macro(topic, country, start, end)")
        print("  pipeline.get_entity(entity, start, end)")
