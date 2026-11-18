
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings


def create_mcp_client():
    config = {
        "transport": settings.stateinfo_mcp_transport,
        "url": settings.stateinfo_mcp_url,
    }

    if settings.stateinfo_mcp_token:
        config["headers"] = {
            "Authorization": f"Bearer {settings.stateinfo_mcp_token}"
        }

    return MultiServerMCPClient({
        "stateinfo": config
    })


async def discover_mcp_tools(client):
    tools = await client.get_tools()

    if not tools:
        raise RuntimeError(
            "StateInfoMCP returned no tools"
        )

    return tools
