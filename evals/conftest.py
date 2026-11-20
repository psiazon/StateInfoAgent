import os
import uuid

import httpx
import pytest

AGENT_URL = os.getenv(
    "AGENT_EVAL_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


@pytest.fixture(scope="session")
def agent_url():
    return AGENT_URL


@pytest.fixture
def thread_id():
    return f"eval-{uuid.uuid4()}"


@pytest.fixture
def chat(agent_url):
    def send(message, thread_id=None):
        payload = {
            "message": message,
            "thread_id": thread_id or str(uuid.uuid4())
        }

        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                f"{agent_url}/api/agent/chat",
                json=payload
            )

        response.raise_for_status()
        return response.json()

    return send