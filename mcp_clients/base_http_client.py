# ==========================================================================================================
# Imports
# ==========================================================================================================

import logging
import sys

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters         # For both STDIO and HTTP Clients
from mcp.client.stdio import stdio_client                    # For STDIO Client
from mcp.client.streamable_http import streamablehttp_client # For HTTP Client



logging.getLogger("mcp").setLevel(logging.WARNING)


# ==========================================================================================================
# Base HTTP Client
# ==========================================================================================================

class MCPHTTPClient:
    """
        Base MCP HTTP client with pure protocol logic - no GUI dependencies.
    """

    # MCP set-up methods
    def __init__(self, server_url: str, roots_dir: str):
        self.server_url = server_url
        self.roots_dir = roots_dir
        self.session = None
        self.exit_stack = AsyncExitStack()
        self._connected = False

    # MCP Handshake method
    async def connect(self):
        if self._connected:
            return

        mcp_url = f"{self.server_url}/mcp" # FastMCP uses the endpoint `/mcp` for streamable HTTP by default.
        read, write, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(mcp_url)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()
        self._connected = True

    # MCP Tools methods
    async def list_tools(self):
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, arguments: dict):
        result = await self.session.call_tool(tool_name, arguments)
        return result

    # MCP Resources methods
    async def list_resources(self):
        result = await self.session.list_resource_templates()
        return result.resourceTemplates

    async def read_resource(self, uri: str):
        result = await self.session.read_resource(uri)
        return result

    # MCP Prompts methods
    async def list_prompts(self):
        result = await self.session.list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, arguments: dict):
        result = await self.session.get_prompt(prompt_name, arguments)
        return result

    # MCP Clean-up method
    async def cleanup(self):
        """Clean up resources and close HTTP connection."""
        await self.exit_stack.aclose()