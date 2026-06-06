from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    POSTGRES_USER: str = "tfg"
    POSTGRES_PASSWORD: str = "changeme"
    POSTGRES_DB: str = "tfg_experiments"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    # Mongo
    MONGO_URI: str = "mongodb://mongo:27017/tfg_news"
    MONGO_DB: str = "tfg_news"

    # JWT
    JWT_SECRET: str = "changeme-dev"
    JWT_ALG: str = "HS256"
    JWT_EXPIRES_MIN: int = 10080  # 7 days

    # Admin seed (created on first startup if DB is empty)
    ADMIN_EMAIL: str = "admin@tfg.local"
    ADMIN_PASSWORD: str = "changeme-dev"

    # MCP server
    MCP_SERVER_URL: str = "http://mcp_server:8080/sse"

    # Ollama narration (desktop Ollama reachable via Docker host gateway)
    OLLAMA_URL: str = "http://host.docker.internal:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    # Separate model for the in-app chat tutor. Default = llama3.2:3b because
    # qwen3:4b on this Ollama runtime ignores both `think:false` and `/no_think`
    # and streams its chain-of-thought as visible content (bad UX). Override via
    # OLLAMA_CHAT_MODEL=qwen3:8b in .env once a non-thinking qwen3 is installed.
    OLLAMA_CHAT_MODEL: str = "llama3.2:3b"

    # MLflow tracking
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"

    # CORS — comma-separated origins (empty = allow all, dev default)
    CORS_ORIGINS: str = ""

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
