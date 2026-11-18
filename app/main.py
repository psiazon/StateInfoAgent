
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage

from app.config import settings
from app.mcp_client import (
    create_mcp_client,
    discover_mcp_tools,
)
from app.agent import build_agent
from app.schemas import ChatRequest, ChatResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StateInfoAgent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing StateInfoAgent")

    client = create_mcp_client()
    tools = await discover_mcp_tools(client)

    app.state.agent = build_agent(tools)
    app.state.tool_names = [tool.name for tool in tools]

    logger.info(
        "Loaded %s MCP tools",
        len(tools),
    )

    yield

    logger.info("StateInfoAgent shutting down")


app = FastAPI(
    title="StateInfoAgent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "StateInfoAgent",
    }


@app.get("/api/agent/tools")
async def get_tools():
    return {
        "tools": app.state.tool_names
    }


@app.post(
    "/api/agent/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    try:
        result = await app.state.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request.message,
                    }
                ]
            },
            config={
                "recursion_limit": settings.agent_max_steps
            },
        )

        messages = result["messages"]

        tools_used = [
            call["name"]
            for msg in messages
            if isinstance(msg, AIMessage)
            for call in msg.tool_calls
        ]

        final_message = messages[-1]

        content = final_message.content
        if isinstance(content, str):
            answer = content
        else:
            answer = str(content)

        return ChatResponse(
            response=answer,
            tools_used=tools_used,
        )

    except Exception:
        logger.exception("Agent execution failed")

        raise HTTPException(
            status_code=502,
            detail="Agent execution failed"
        )
