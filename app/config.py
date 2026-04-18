from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        protected_namespaces=("settings_",),
    )

    anthropic_api_key: str
    mongo_uri: str = "mongodb://mongo:27017"
    mongo_db: str = "codegraph"
    model_dir: str = "./models_cache"
    faiss_dir: str = "./faiss_indexes"
    gnn_checkpoint: str = "./models_cache/gnn_checkpoint.pt"
    embedding_model: str = "all-mpnet-base-v2"
    top_k_rag: int = 8
    gnn_suspect_threshold: float = 0.65

settings = Settings()

# Ensure directories exist
Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
Path(settings.faiss_dir).mkdir(parents=True, exist_ok=True)
