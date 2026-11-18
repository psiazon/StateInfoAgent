from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from langchain_openai import ChatOpenAI

from app.config import settings


def create_llm():
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return ChatOpenAI(
        model=settings.azure_openai_deployment,
        base_url=settings.azure_openai_endpoint.rstrip("/") + "/",
        api_key=token_provider,
        temperature=0,
        max_retries=2,
        timeout=60,
    )