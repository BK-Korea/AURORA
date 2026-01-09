"""Configuration management for AURORA."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    glm_api_key: str = Field(..., env="GLM_API_KEY")
    glm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        env="GLM_BASE_URL"
    )

    # Model Configuration
    chat_model: str = Field(default="glm-4.7", env="CHAT_MODEL")
    embedding_model: str = Field(default="embedding-3", env="EMBEDDING_MODEL")

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default=None)
    raw_dir: Path = Field(default=None)
    processed_dir: Path = Field(default=None)
    vectordb_dir: Path = Field(default=None)

    # RAG Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=10, env="RETRIEVAL_TOP_K")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set derived paths
        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.vectordb_dir = self.data_dir / "vectordb"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vectordb_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
