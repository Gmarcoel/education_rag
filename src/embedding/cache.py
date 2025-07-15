import pickle
import hashlib
from pathlib import Path
import numpy as np

from config.settings import Settings


class EmbeddingCache:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = Path(settings.EMBEDDINGS_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(self, content: str, model_name: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        model_hash = hashlib.md5(model_name.encode()).hexdigest()
        return f"{content_hash}_{model_hash}"
    
    def save_embedding(self, content: str, model_name: str, embedding: np.ndarray):
        cache_key = self.get_cache_key(content, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
    
    def load_embedding(self, content: str, model_name: str) -> np.ndarray | None:
        cache_key = self.get_cache_key(content, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def save_batch_embeddings(self, documents: list[dict[str, object]], model_name: str):
        for document in documents:
            if 'embedding' in document:
                embedding = np.array(document['embedding'])
                self.save_embedding(document['content'], model_name, embedding)
    
    def load_batch_embeddings(self, documents: list[dict[str, object]], model_name: str) -> list[dict[str, object]]:
        cached_documents = []
        
        for document in documents:
            cached_embedding = self.load_embedding(document['content'], model_name)
            
            if cached_embedding is not None:
                cached_doc = document.copy()
                cached_doc['embedding'] = cached_embedding.tolist()
                cached_doc['metadata']['cached'] = True
                cached_documents.append(cached_doc)
            else:
                cached_documents.append(document)
        
        return cached_documents
    
    def clear_cache(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
    
    def get_cache_size(self) -> int:
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
