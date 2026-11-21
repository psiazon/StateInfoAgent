import json
import uuid
import logging
from contextlib import asynccontextmanager
from langgraph.checkpoint.memory import InMemorySaver
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

    # In-memory mode is for local testing only; Azure deployments should use postgres.
    if settings.memory_backend == "postgres":
        if not settings.postgres_connection_string:
            raise RuntimeError("POSTGRES_CONNECTION_STRING is required for postgres memory")
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(settings.postgres_connection_string) as checkpointer:
            await checkpointer.setup()
            app.state.agent = build_agent(tools, checkpointer)
            app.state.tool_names = [tool.name for tool in tools]
            yield
    elif settings.memory_backend == "memory":
        app.state.agent = build_agent(tools, InMemorySaver())
        app.state.tool_names = [tool.name for tool in tools]
        yield
    else:
        raise RuntimeError("MEMORY_BACKEND must be memory or postgres")

    logger.info("StateInfoAgent shutting down")

    # Startup and shutdown handled above.
    return



app = FastAPI(
    title="StateInfoAgent",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow StateInfoChatBot to communicate with StateInfoAgent
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local React development
        "http://localhost:5173",

        # Production Azure StateInfoChatBot
        "https://stateinfochatbot-ebhweeanakf3bgc3.westcentralus-01.azurewebsites.net",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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
        thread_id = request.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.agent_max_steps}
        prior_state = await app.state.agent.aget_state(config)
        prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

        result = await app.state.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request.message,
                    }
                ]
            },
            config=config,
        )

        messages = result["messages"]

        # Only report tools and citations from this turn, not the whole saved thread.
        current_messages = messages[prior_count:]

        tools_used = [
            call["name"]
            for msg in current_messages
            if isinstance(msg, AIMessage)
            for call in msg.tool_calls
        ]

        citations = extract_citations(current_messages)

        final_message = messages[-1]

        content = final_message.content
        if isinstance(content, str):
            answer = content
        else:
            answer = str(content)

        return ChatResponse(
            thread_id=thread_id,
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
