
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    azure_openai_endpoint: str
    azure_openai_deployment: str = "gpt-4.1-mini"
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_auth_mode: str = "entra"
    azure_openai_api_key: str | None = None

    stateinfo_mcp_url: str
    stateinfo_mcp_transport: str = "http"
    stateinfo_mcp_token: str | None = None

    azure_search_endpoint: str | None = None
    azure_search_index_name: str = "rag-1795046232229"
    azure_search_vector_field: str = "text_vector"
    azure_search_content_field: str = "chunk"
    azure_search_title_field: str = "title"
    azure_search_k: int = 30
    azure_search_top: int = 30

    memory_backend: str = "memory"  # memory or postgres
    postgres_connection_string: str | None = None

    agent_max_steps: int = 12

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
