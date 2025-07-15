from dataclasses import dataclass, field
import time
import json
from pathlib import Path

from config.settings import Settings
from src.evaluation.metrics import EvaluationMetric, RetrievalMetric, EvaluationResult


@dataclass
class EvaluationReport:
    timestamp: str
    total_queries: int
    avg_execution_time: float
    metric_results: dict[str, float] = field(default_factory=dict)
    detailed_results: list[dict[str, object]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "avg_execution_time": self.avg_execution_time,
            "metric_results": self.metric_results,
            "detailed_results": self.detailed_results
        }
    
    def save(self, filepath: Path):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class RAGEvaluator:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.generation_metrics: list[EvaluationMetric] = []
        self.retrieval_metrics: list[RetrievalMetric] = []
    
    def add_generation_metric(self, metric: EvaluationMetric):
        self.generation_metrics.append(metric)
    
    def add_retrieval_metric(self, metric: RetrievalMetric):
        self.retrieval_metrics.append(metric)
    
    def evaluate_generation(self, predictions: list[str], references: list[str], 
                          contexts: list[str] | None = None) -> dict[str, float]:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        results = {}
        
        for metric in self.generation_metrics:
            metric_scores = []
            
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                context = contexts[i] if contexts else None
                evaluation_result = metric.evaluate(pred, ref, context)
                metric_scores.append(evaluation_result.score)
            
            results[metric.get_name()] = sum(metric_scores) / len(metric_scores)
        
        return results
    
    def evaluate_retrieval(self, retrieved_results: list[list[dict[str, object]]], 
                          relevant_docs: list[list[str]]) -> dict[str, float]:
        if len(retrieved_results) != len(relevant_docs):
            raise ValueError("Retrieved results and relevant docs must have the same length")
        
        results = {}
        
        for metric in self.retrieval_metrics:
            metric_scores = []
            
            for retrieved, relevant in zip(retrieved_results, relevant_docs):
                evaluation_result = metric.evaluate(retrieved, relevant)
                metric_scores.append(evaluation_result.score)
            
            results[metric.get_name()] = sum(metric_scores) / len(metric_scores)
        
        return results
    
    def evaluate_end_to_end(self, queries: list[str], predictions: list[str], 
                           references: list[str], retrieved_docs: list[list[dict[str, object]]],
                           relevant_docs: list[list[str]], 
                           contexts: list[str] | None = None) -> EvaluationReport:
        start_time = time.time()
        
        generation_results = self.evaluate_generation(predictions, references, contexts)
        retrieval_results = self.evaluate_retrieval(retrieved_docs, relevant_docs)
        
        execution_time = time.time() - start_time
        
        all_results = {**generation_results, **retrieval_results}
        
        detailed_results = []
        for i, query in enumerate(queries):
            detailed_result = {
                "query": query,
                "prediction": predictions[i],
                "reference": references[i],
                "context": contexts[i] if contexts else None,
                "retrieved_docs": len(retrieved_docs[i]),
                "relevant_docs": len(relevant_docs[i])
            }
            detailed_results.append(detailed_result)
        
        report = EvaluationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_queries=len(queries),
            avg_execution_time=execution_time / len(queries),
            metric_results=all_results,
            detailed_results=detailed_results
        )
        
        return report
    
    def evaluate_single_query(self, query: str, prediction: str, reference: str,
                            retrieved_docs: list[dict[str, object]], 
                            relevant_docs: list[str],
                            context: str | None = None) -> dict[str, EvaluationResult]:
        results = {}
        
        for metric in self.generation_metrics:
            result = metric.evaluate(prediction, reference, context)
            results[metric.get_name()] = result
        
        for metric in self.retrieval_metrics:
            result = metric.evaluate(retrieved_docs, relevant_docs)
            results[metric.get_name()] = result
        
        return results
