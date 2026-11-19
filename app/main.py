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
from app.search_tool import create_search_tool
from app.schemas import ChatRequest, ChatResponse, Citation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StateInfoAgent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing StateInfoAgent")

    client = create_mcp_client()
    tools = await discover_mcp_tools(client)

    search_tool = create_search_tool()
    if search_tool is not None:
        tools.append(search_tool)
        logger.info("Azure AI Search hybrid retrieval enabled")
    else:
        logger.info("Azure AI Search not configured; MCP-only mode")

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
    citations = []
    seen = set()

    # Map tool call IDs to tool names.
    call_details = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                call_id = call.get("id")
                if call_id:
                    call_details[call_id] = call.get("name")

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        tool_name = (
            call_details.get(message.tool_call_id)
            or message.name
            or "Unknown tool"
        )

        logger.info(
            "Processing citation | tool=%s | call_id=%s",
            tool_name,
            message.tool_call_id,
        )

        # Azure AI Search citations
        if tool_name == "search_company_documents":
            try:
                raw_content = message.content

                if isinstance(raw_content, str):
                    payload = json.loads(raw_content)

                elif isinstance(raw_content, list):
                    text = "\n".join(
                        item.get("text", "")
                        if isinstance(item, dict)
                        else item
                        for item in raw_content
                        if isinstance(item, (dict, str))
                    )
                    payload = json.loads(text)

                elif isinstance(raw_content, dict):
                    payload = raw_content

                else:
                    logger.warning(
                        "Unsupported search result type: %s",
                        type(raw_content).__name__,
                    )
                    continue

                if not isinstance(payload, dict):
                    logger.warning(
                        "Search result is not a JSON object"
                    )
                    continue

                if payload.get("error"):
                    logger.warning(
                        "Search tool returned error: %s",
                        payload["error"],
                    )
                    continue

                documents = payload.get("results", [])

                logger.info(
                    "Found %d search citation candidates",
                    len(documents),
                )

                for doc in documents:
                    if not isinstance(doc, dict):
                        continue

                    title = doc.get("title")
                    chunk_id = doc.get("chunk_id")

                    if not title:
                        continue

                    key = (title, chunk_id)

                    if key in seen:
                        continue

                    seen.add(key)

                    citations.append(
                        Citation(
                            title=title,
                            source="Azure AI Search",
                            url=None,
                            tool=tool_name,
                            snippet=str(
                                doc.get("content", "")
                            )[:500] or None,
                        )
                    )

            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Unable to parse Azure AI Search citations: %s",
                    exc,
                )

        # MCP citations
        else:
            key = (
                "mcp",
                message.tool_call_id,
                tool_name,
            )

            if key in seen:
                continue

            seen.add(key)

            citations.append(
                Citation(
                    title=f"{tool_name} result",
                    source="StateInfoMCP",
                    url=settings.stateinfo_mcp_url,
                    tool=tool_name,
                    snippet=extract_citation_snippet(
                        message.content
                    ) or None,
                )
            )

    logger.info(
        "Extracted %d citations",
        len(citations),
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
