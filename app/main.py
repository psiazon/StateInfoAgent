import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, ToolMessage

from app.config import settings
from app.mcp_client import (
    create_mcp_client,
    discover_mcp_tools,
)
from app.agent import build_agent
from app.schemas import ChatRequest, ChatResponse, Citation


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


def extract_citation_snippet(content) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        text = "\n".join(text_parts)
        if not text:
            text = json.dumps(content, ensure_ascii=False)
    else:
        text = json.dumps(content, ensure_ascii=False)

    return text[:500] if text else ""


def extract_citations(messages) -> list[Citation]:
    call_details = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                call_details[call.get("id")] = call

    citations = []
    seen = set()

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        tool_name = message.name or call_details.get(
            message.tool_call_id,
            {},
        ).get("name", "MCP tool")

        key = (message.tool_call_id, tool_name)
        if key in seen:
            continue
        seen.add(key)

        snippet = extract_citation_snippet(message.content)
        citations.append(
            Citation(
                title=f"{tool_name} result",
                source="StateInfoMCP",
                url=settings.stateinfo_mcp_url,
                tool=tool_name,
                snippet=snippet or None,
            )
        )

    return citations


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

        citations = extract_citations(messages)

        final_message = messages[-1]

        content = final_message.content
        if isinstance(content, str):
            answer = content
        else:
            answer = str(content)

        return ChatResponse(
            response=answer,
            tools_used=tools_used,
            citations=citations,
        )

    except Exception:
        logger.exception("Agent execution failed")

        raise HTTPException(
            status_code=502,
            detail="Agent execution failed"
        )
