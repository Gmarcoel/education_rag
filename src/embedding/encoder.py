from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import Settings


class EmbeddingModel(ABC):
    
    @abstractmethod
    def encode(self, texts: str | list[str]) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass


class SentenceTransformerEncoder(EmbeddingModel):
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class DocumentEncoder:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self._create_model()
    
    def _create_model(self) -> EmbeddingModel:
        return SentenceTransformerEncoder(self.settings.EMBEDDING_MODEL)
    
    def encode_documents(self, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts)
        
        encoded_documents = []
        for i, document in enumerate(documents):
            encoded_doc = document.copy()
            encoded_doc['embedding'] = embeddings[i].tolist()
            encoded_doc['metadata']['embedding_model'] = self.settings.EMBEDDING_MODEL
            encoded_doc['metadata']['embedding_dimension'] = self.model.get_dimension()
            encoded_documents.append(encoded_doc)
        
        return encoded_documents
    
    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]
