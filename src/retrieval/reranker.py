from abc import ABC, abstractmethod
import numpy as np

from config.settings import Settings


class Reranker(ABC):
    
    @abstractmethod
    def rerank(self, query: str, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        pass


class DistanceReranker(Reranker):
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def rerank(self, query: str, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(documents, key=lambda x: x.get('distance', float('inf')))


class ContentLengthReranker(Reranker):
    
    def __init__(self, settings: Settings, prefer_longer: bool = True):
        self.settings = settings
        self.prefer_longer = prefer_longer
    
    def rerank(self, query: str, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(documents, 
                     key=lambda x: len(x['content']), 
                     reverse=self.prefer_longer)


class DiversityReranker(Reranker):
    
    def __init__(self, settings: Settings, diversity_threshold: float = 0.8):
        self.settings = settings
        self.diversity_threshold = diversity_threshold
    
    def rerank(self, query: str, documents: list[dict[str, object]]) -> list[dict[str, object]]:
        if not documents:
            return documents
        
        reranked = [documents[0]]
        
        for candidate in documents[1:]:
            if self._is_diverse_enough(candidate, reranked):
                reranked.append(candidate)
        
        return reranked
    
    def _is_diverse_enough(self, candidate: dict[str, object], selected: list[dict[str, object]]) -> bool:
        candidate_words = set(candidate['content'].lower().split())
        
        for selected_doc in selected:
            selected_words = set(selected_doc['content'].lower().split())
            similarity = len(candidate_words.intersection(selected_words)) / len(candidate_words.union(selected_words))
            
            if similarity > self.diversity_threshold:
                return False
        
        return True


class RerankerFactory:
    
    @staticmethod
    def create_reranker(strategy: str, settings: Settings, **kwargs) -> Reranker:
        strategies = {
            'distance': DistanceReranker,
            'content_length': ContentLengthReranker,
            'diversity': DiversityReranker
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown reranking strategy: {strategy}")
        
        return strategies[strategy](settings, **kwargs)
