
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = Field(
        default=None,
        description="Conversation ID used by LangGraph memory"
    )

class Citation(BaseModel):
    title: str
    source: str
    url: str | None = None
    tool: str
    snippet: str | None = None

class ChatResponse(BaseModel):
    thread_id: str | None = None
    response: str
    tools_used: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
