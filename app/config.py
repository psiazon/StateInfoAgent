
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

    agent_max_steps: int = 12

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
