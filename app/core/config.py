from pydantic_settings import BaseSettings, SettingsConfigDict

# Settings class jo configuration load karegi environment variables ya .env file se
class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuClip API"
    # DATABASE_URL is the generic setting used by deployments. Keep
    # POSTGRES_URL as a backwards-compatible alias for existing .env files.
    DATABASE_URL: str = ""
    POSTGRES_URL: str = "postgresql://postgres:postgres@localhost:5433/docuclip"
    REDIS_URL: str = "redis://localhost:6379/0"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    WHISPER_MODEL: str = "tiny"
    MAX_QUALITY: int = 720

    # SettingsConfigDict configure kar raha hai pydantic v2 settings load configuration ko
    # Yeh automatically .env file se keys read karega
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Settings instantiate ho rahi hain globally reuse karne ke liye
settings = Settings()
