from abc import ABC, abstractmethod
import numpy as np
import time

from config.settings import Settings
from src.storage.database import VectorDatabase
from src.storage.schemas import SearchResult, QueryResult, Document
from src.embedding.encoder import DocumentEncoder


class RetrievalStrategy(ABC):
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        pass


class SemanticRetriever(RetrievalStrategy):
    
    def __init__(self, database: VectorDatabase, encoder: DocumentEncoder, settings: Settings):
        self.database = database
        self.encoder = encoder
        self.settings = settings
    
    def retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        start_time = time.time()
        
        query_embedding = self.encoder.encode_query(query)
        documents = self.database.search(query_embedding, top_k)
        
        execution_time = time.time() - start_time
        
        return {
            'query': query,
            'documents': documents,
            'execution_time': execution_time,
            'total_results': len(documents)
        }


class HybridRetriever(RetrievalStrategy):
    
    def __init__(self, database: VectorDatabase, encoder: DocumentEncoder, settings: Settings):
        self.database = database
        self.encoder = encoder
        self.settings = settings
    
    def retrieve(self, query: str, top_k: int) -> list[dict[str, object]]:
        start_time = time.time()
        
        semantic_results = self._semantic_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        combined_results = self._combine_results(semantic_results, keyword_results, top_k)
        
        execution_time = time.time() - start_time
        
        return {
            'query': query,
            'documents': combined_results,
            'execution_time': execution_time,
            'total_results': len(combined_results)
        }
    
    def _semantic_search(self, query: str, top_k: int) -> list[dict[str, object]]:
        query_embedding = self.encoder.encode_query(query)
        return self.database.search(query_embedding, top_k)
    
    def _keyword_search(self, query: str, top_k: int) -> list[dict[str, object]]:
        all_documents = self.database.get_all_documents()
        query_terms = query.lower().split()
        
        scored_documents = []
        for doc in all_documents:
            score = self._calculate_keyword_score(doc['content'].lower(), query_terms)
            if score > 0:
                doc_copy = doc.copy()
                doc_copy['keyword_score'] = score
                scored_documents.append(doc_copy)
        
        scored_documents.sort(key=lambda x: x['keyword_score'], reverse=True)
        return scored_documents[:top_k]
    
    def _calculate_keyword_score(self, content: str, query_terms: list[str]) -> float:
        score = 0.0
        content_words = content.split()
        
        for term in query_terms:
            term_count = content.count(term)
            if term_count > 0:
                tf = term_count / len(content_words)
                score += tf
        
        return score
    
    def _combine_results(self, semantic_results: list[dict[str, object]], 
                        keyword_results: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
        combined = {}
        
        for i, doc in enumerate(semantic_results):
            doc_id = doc['id']
            semantic_score = 1.0 / (i + 1)
            combined[doc_id] = {
                'document': doc,
                'semantic_score': semantic_score,
                'keyword_score': 0.0
            }
        
        for i, doc in enumerate(keyword_results):
            doc_id = doc['id']
            keyword_score = doc.get('keyword_score', 0.0)
            
            if doc_id in combined:
                combined[doc_id]['keyword_score'] = keyword_score
            else:
                combined[doc_id] = {
                    'document': doc,
                    'semantic_score': 0.0,
                    'keyword_score': keyword_score
                }
        
        for doc_id in combined:
            semantic_weight = 0.7
            keyword_weight = 0.3
            combined[doc_id]['final_score'] = (
                semantic_weight * combined[doc_id]['semantic_score'] +
                keyword_weight * combined[doc_id]['keyword_score']
            )
        
        sorted_results = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
        return [result['document'] for result in sorted_results[:top_k]]


class RetrieverFactory:
    
    @staticmethod
    def create_retriever(strategy: str, database: VectorDatabase, 
                        encoder: DocumentEncoder, settings: Settings) -> RetrievalStrategy:
        strategies = {
            'semantic': SemanticRetriever,
            'hybrid': HybridRetriever
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        
        return strategies[strategy](database, encoder, settings)
