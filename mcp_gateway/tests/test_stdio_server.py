"""Real MCP transport smoke; every operation stays local and read-only."""
import asyncio
import json
from pathlib import Path
import sys
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class StdioGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_tools_are_discoverable_and_ambient_write_token_is_removed(self):
        root = Path(__file__).resolve().parents[2]
        parameters = StdioServerParameters(
            command=sys.executable, args=["-P", "-m", "mcp_gateway.stdio_server"],
            env={"PYTHONPATH": str(root), "PYTHONNOUSERSITE": "1",
                 "JANUS_ALLOW_WRITES": "1", "GITHUB_TOKEN": "synthetic-never-used"},
            cwd=root,
        )

        async def check():
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    self.assertIn("janus.codex_cloud_handoff", names)
                    self.assertIn("janus.codex_app_server_readiness", names)
                    self.assertNotIn("janus.codex_bridge_login", names)
                    health = await session.call_tool("janus.health", {})
                    data = health.structured_content
                    if data is None:
                        data = json.loads(health.content[0].text)
                    self.assertFalse(data["write_enabled"])
                    self.assertFalse(data["github_token_present"])
                    readiness = await session.call_tool("janus.codex_app_server_readiness", {})
                    data = readiness.structured_content
                    if data is None:
                        data = json.loads(readiness.content[0].text)
                    self.assertFalse(data["live_account_checked"])
                    self.assertFalse(data["oauth_token_accessed"])

        await asyncio.wait_for(check(), timeout=20)


if __name__ == "__main__":
    unittest.main()
