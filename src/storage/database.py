from abc import ABC, abstractmethod
import chromadb
import numpy as np

from config.settings import Settings


class VectorDatabase(ABC):
    
    @abstractmethod
    def add_documents(self, documents: list[dict[str, object]]):
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, object]]:
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: list[str]):
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        pass


class ChromaVectorDatabase(VectorDatabase):
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = chromadb.PersistentClient(path=settings.DATABASE_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.DATABASE_COLLECTION
        )
    
    def add_documents(self, documents: list[dict[str, object]]):
        if not documents:
            return
        
        ids = [doc['id'] for doc in documents]
        embeddings = [doc['embedding'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, object]]:
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                document = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                documents.append(document)
        
        return documents
    
    def delete_documents(self, document_ids: list[str]):
        if document_ids:
            self.collection.delete(ids=document_ids)
    
    def get_document_count(self) -> int:
        return self.collection.count()
    
    def get_all_documents(self) -> list[dict[str, object]]:
        results = self.collection.get()
        documents = []
        
        if results['documents']:
            for i in range(len(results['documents'])):
                document = {
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                }
                documents.append(document)
        
        return documents
    
    def clear_collection(self):
        self.client.delete_collection(self.settings.DATABASE_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.DATABASE_COLLECTION
        )


class DatabaseManager:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = self._create_database()
    
    def _create_database(self) -> VectorDatabase:
        return ChromaVectorDatabase(self.settings)
    
    def __enter__(self):
        return self.database
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
