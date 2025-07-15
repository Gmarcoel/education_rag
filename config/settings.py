from dataclasses import dataclass


@dataclass
class Settings:
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K: int = 5
    LLM_MODEL: str = "gemini-1.5-flash"
    DATABASE_COLLECTION: str = "documents"
    DATABASE_PERSIST_DIR: str = "persistent/"
    DATA_RAW_DIR: str = "data/raw/"
    DATA_PROCESSED_DIR: str = "data/processed/"
    EMBEDDINGS_CACHE_DIR: str = "data/embeddings/"
    API_KEY_FILE: str = ".api_key"
    MAX_DOCUMENTS: int | None = None
