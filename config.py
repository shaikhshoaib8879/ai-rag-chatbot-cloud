from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    temperature: float = 0.3

    @property
    def chroma_db_path(self) -> str:
        return str(Path(__file__).parent / "chroma_db")


settings = Settings()
