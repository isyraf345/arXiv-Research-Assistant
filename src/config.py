"""
Configuration settings for arXiv RAG Assistant
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Database
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/arxiv_rag",
        env="DATABASE_URL"
    )
    
    # ChromaDB
    chroma_persist_directory: str = Field(
        default="./chroma_db",
        env="CHROMA_PERSIST_DIRECTORY"
    )
    
    # Embeddings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Text Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Retrieval
    retrieval_k: int = Field(default=5, env="RETRIEVAL_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # LLM
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=1000, env="MAX_TOKENS")
    
    # Data Storage
    papers_storage_path: str = Field(
        default="./data/papers",
        env="PAPERS_STORAGE_PATH"
    )
    
    # arXiv API
    arxiv_max_results: int = Field(default=100, env="ARXIV_MAX_RESULTS")
    arxiv_delay: float = Field(default=3.0, env="ARXIV_DELAY")  # Rate limiting
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
def setup_directories():
    """Create required directories if they don't exist"""
    Path(settings.papers_storage_path).mkdir(parents=True, exist_ok=True)
    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)

# Validation
def validate_config():
    """Validate critical configuration"""
    if not settings.openai_api_key or settings.openai_api_key == "your-api-key-here":
        raise ValueError("OpenAI API key not set. Please set OPENAI_API_KEY in .env file")
    
    return True

if __name__ == "__main__":
    setup_directories()
    validate_config()
    print("✅ Configuration validated successfully")