from abc import ABC, abstractmethod
import time
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    max_score: float
    details: dict[str, object] | None = None


class EvaluationMetric(ABC):
    
    @abstractmethod
    def evaluate(self, predicted: str, reference: str, context: str | None = None) -> EvaluationResult:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class RetrievalMetric(ABC):
    
    @abstractmethod
    def evaluate(self, retrieved_docs: list[dict[str, object]], 
                relevant_docs: list[str]) -> EvaluationResult:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class BLEUMetric(EvaluationMetric):
    
    def evaluate(self, predicted: str, reference: str, context: str | None = None) -> EvaluationResult:
        pred_tokens = predicted.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return EvaluationResult("BLEU", 0.0, 1.0)
        
        common_tokens = set(pred_tokens).intersection(set(ref_tokens))
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return EvaluationResult(
            metric_name="BLEU",
            score=f1_score,
            max_score=1.0,
            details={
                "precision": precision,
                "recall": recall,
                "common_tokens": len(common_tokens),
                "predicted_tokens": len(pred_tokens),
                "reference_tokens": len(ref_tokens)
            }
        )
    
    def get_name(self) -> str:
        return "BLEU"


class SemanticSimilarityMetric(EvaluationMetric):
    
    def __init__(self, encoder=None):
        self.encoder = encoder
    
    def evaluate(self, predicted: str, reference: str, context: str | None = None) -> EvaluationResult:
        if not self.encoder:
            return EvaluationResult("Semantic_Similarity", 0.0, 1.0, 
                                   details={"error": "No encoder provided"})
        
        pred_embedding = self.encoder.encode_query(predicted)
        ref_embedding = self.encoder.encode_query(reference)
        
        similarity = np.dot(pred_embedding, ref_embedding) / (
            np.linalg.norm(pred_embedding) * np.linalg.norm(ref_embedding)
        )
        
        return EvaluationResult(
            metric_name="Semantic_Similarity",
            score=float(similarity),
            max_score=1.0,
            details={"cosine_similarity": float(similarity)}
        )
    
    def get_name(self) -> str:
        return "Semantic_Similarity"


class RelevanceMetric(EvaluationMetric):
    
    def evaluate(self, predicted: str, reference: str, context: str | None = None) -> EvaluationResult:
        if not context:
            return EvaluationResult("Relevance", 0.0, 1.0, 
                                   details={"error": "No context provided"})
        
        pred_words = set(predicted.lower().split())
        context_words = set(context.lower().split())
        
        overlap = len(pred_words.intersection(context_words))
        relevance_score = overlap / len(pred_words) if pred_words else 0.0
        
        return EvaluationResult(
            metric_name="Relevance",
            score=relevance_score,
            max_score=1.0,
            details={
                "context_overlap": overlap,
                "predicted_words": len(pred_words),
                "context_words": len(context_words)
            }
        )
    
    def get_name(self) -> str:
        return "Relevance"


class PrecisionAtK(RetrievalMetric):
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def evaluate(self, retrieved_docs: list[dict[str, object]], 
                relevant_docs: list[str]) -> EvaluationResult:
        if not retrieved_docs:
            return EvaluationResult(f"Precision@{self.k}", 0.0, 1.0)
        
        top_k_docs = retrieved_docs[:self.k]
        retrieved_ids = [doc['id'] for doc in top_k_docs]
        
        relevant_retrieved = len(set(retrieved_ids).intersection(set(relevant_docs)))
        precision = relevant_retrieved / len(top_k_docs)
        
        return EvaluationResult(
            metric_name=f"Precision@{self.k}",
            score=precision,
            max_score=1.0,
            details={
                "relevant_retrieved": relevant_retrieved,
                "total_retrieved": len(top_k_docs),
                "relevant_documents": len(relevant_docs)
            }
        )
    
    def get_name(self) -> str:
        return f"Precision@{self.k}"


class RecallAtK(RetrievalMetric):
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def evaluate(self, retrieved_docs: list[dict[str, object]], 
                relevant_docs: list[str]) -> EvaluationResult:
        if not relevant_docs:
            return EvaluationResult(f"Recall@{self.k}", 0.0, 1.0)
        
        top_k_docs = retrieved_docs[:self.k]
        retrieved_ids = [doc['id'] for doc in top_k_docs]
        
        relevant_retrieved = len(set(retrieved_ids).intersection(set(relevant_docs)))
        recall = relevant_retrieved / len(relevant_docs)
        
        return EvaluationResult(
            metric_name=f"Recall@{self.k}",
            score=recall,
            max_score=1.0,
            details={
                "relevant_retrieved": relevant_retrieved,
                "total_relevant": len(relevant_docs),
                "retrieved_documents": len(top_k_docs)
            }
        )
    
    def get_name(self) -> str:
        return f"Recall@{self.k}"
