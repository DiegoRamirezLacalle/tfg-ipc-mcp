"""Synchronous MCP client that launches mcp_server.py as a stdio subprocess.

Uses a dedicated thread for the async event loop to avoid cancel-scope
conflicts between asyncio and anyio (Python 3.10).

Usage:
    from mcp_client import MCPPipeline

    with MCPPipeline() as p:
        p.fetch_gdelt("2023-01-01", "2023-12-31")
        p.fetch_rss("bce", "2023-01-01", "2023-12-31")

Requirements:
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from shared.logger import get_logger

logger = get_logger(__name__)

SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"
PYTHON_EXE    = sys.executable


class MCPPipeline:
    """Synchronous wrapper over the async MCP client.

    The event loop runs in a dedicated thread (run_coroutine_threadsafe pattern)
    to avoid anyio cancel-scope conflicts in Python 3.10.
    """

    def __init__(self, timeout: int = 120):
        self._timeout   = timeout
        self._loop      = asyncio.new_event_loop()
        self._thread    = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._session: ClientSession | None = None
        self._exit_stack = None

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

        tools      = await self._session.list_tools()
        tool_names = [t.name for t in tools.tools]
        logger.info(f"[MCP] Connected. Tools: {tool_names}")

    async def _call_tool(self, name: str, arguments: dict) -> str:
        if self._session is None:
            raise RuntimeError("Session not initialized")
        result = await self._session.call_tool(name, arguments)
        return result.content[0].text

    def _run(self, coro):
        """Execute a coroutine on the event-loop thread and block for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=self._timeout)

    # Public methods

    def fetch_gdelt(self, start_date: str, end_date: str) -> list[dict]:
        """Download GDELT v2 quantitative monthly signals."""
        raw = self._run(self._call_tool("fetch_gdelt_spain", {
            "start_date": start_date,
            "end_date":   end_date,
        }))
        return json.loads(raw)

    def fetch_rss(self, source: str, start_date: str, end_date: str) -> dict:
        """Download RSS from an official source (bce|ine|bde)."""
        raw = self._run(self._call_tool("fetch_rss_official", {
            "source":     source,
            "start_date": start_date,
            "end_date":   end_date,
        }))
        return json.loads(raw)

    def search(self, query: str, start_date: str, end_date: str) -> list[dict]:
        """Search across all stored sources."""
        raw = self._run(self._call_tool("search_news", {
            "query":      query,
            "start_date": start_date,
            "end_date":   end_date,
        }))
        return json.loads(raw)

    def get_macro(self, topic: str, country: str, start_date: str, end_date: str) -> list[dict]:
        """Retrieve macro news by topic."""
        raw = self._run(self._call_tool("get_macro_news", {
            "topic":      topic,
            "country":    country,
            "start_date": start_date,
            "end_date":   end_date,
        }))
        return json.loads(raw)

    def get_entity(self, entity: str, start_date: str, end_date: str) -> list[dict]:
        """Retrieve news by entity (bce|ine|bde)."""
        raw = self._run(self._call_tool("get_entity_news", {
            "entity":     entity,
            "start_date": start_date,
            "end_date":   end_date,
        }))
        return json.loads(raw)

    # Lifecycle

    def close(self):
        """Close the MCP session and event loop."""
        if self._exit_stack:
            future = asyncio.run_coroutine_threadsafe(
                self._exit_stack.aclose(), self._loop
            )
            try:
                future.result(timeout=10)
            except Exception:
                pass  # server already finished, ignore cleanup error

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# CLI test
if __name__ == "__main__":
    logger.info("=== MCP Client test ===")
    with MCPPipeline() as pipeline:
        logger.info("Client OK. Available methods:")
        logger.info("  pipeline.fetch_gdelt(start, end)")
        logger.info("  pipeline.fetch_rss(source, start, end)")
        logger.info("  pipeline.search(query, start, end)")
        logger.info("  pipeline.get_macro(topic, country, start, end)")
        logger.info("  pipeline.get_entity(entity, start, end)")
