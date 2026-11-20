import httpx


def test_agent_health(agent_url):
    response = httpx.get(
        f"{agent_url}/health",
        timeout=15
    )

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_agent_tools_available(agent_url):
    response = httpx.get(
        f"{agent_url}/api/agent/tools",
        timeout=15
    )

    assert response.status_code == 200

    tools = response.json()["tools"]

    assert isinstance(tools, list)
    assert len(tools) > 0