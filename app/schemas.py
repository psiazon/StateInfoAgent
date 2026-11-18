
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000
    )


class Citation(BaseModel):
    title: str
    source: str
    url: str
    tool: str
    snippet: str | None = None


class ChatResponse(BaseModel):
    response: str
    tools_used: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
