from dataclasses import dataclass, field
from typing import Any, Optional
import time
import json
from pathlib import Path
from itertools import product

from config.settings import Settings
from src.evaluation.judge import JudgeFactory, JudgeResult
from src.evaluation.synthetic_dataset import SyntheticQA
from scripts.demo import RAGPipeline


@dataclass
class BenchmarkConfig:
    chunking_strategy: str
    chunk_size: int
    embedding_model: str
    retrieval_strategy: str
    top_k: int
    language: str = "english"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "embedding_model": self.embedding_model,
            "retrieval_strategy": self.retrieval_strategy,
            "top_k": self.top_k,
            "language": self.language
        }
    
    def get_identifier(self) -> str:
        return f"chunk_{self.chunking_strategy}_{self.chunk_size}_emb_{self.embedding_model.replace('/', '_')}_retr_{self.retrieval_strategy}_k{self.top_k}_{self.language}"


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    qa_results: list[dict[str, Any]] = field(default_factory=list)
    avg_scores: dict[str, float] = field(default_factory=dict)
    total_queries: int = 0
    avg_execution_time: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "qa_results": self.qa_results,
            "avg_scores": self.avg_scores,
            "total_queries": self.total_queries,
            "avg_execution_time": self.avg_execution_time,
            "timestamp": self.timestamp
        }


class RAGBenchmark:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        self.judges = [
            JudgeFactory.create_judge("correctness", settings),
            JudgeFactory.create_judge("relevance", settings),
            JudgeFactory.create_judge("groundedness", settings)
        ]
    
    def run_single_config(self, config: BenchmarkConfig, qa_dataset: list[SyntheticQA]) -> BenchmarkResult:
        modified_settings = self._apply_config_to_settings(config)
        pipeline = RAGPipeline(modified_settings)
        
        if pipeline.get_database_stats()["document_count"] == 0:
            pipeline.build_index()
        
        start_time = time.time()
        qa_results = []
        
        for qa in qa_dataset:
            query_start = time.time()
            
            try:
                result = pipeline.query(qa.question, config.top_k)
                query_time = time.time() - query_start
                
                context = self._extract_context_from_result(result)
                
                judge_results = {}
                for judge in self.judges:
                    judge_result = judge.evaluate(qa.question, result['answer'], qa.answer, context)
                    judge_results[judge_result.criteria] = {
                        'score': judge_result.score,
                        'rationale': judge_result.rationale
                    }
                
                qa_result = {
                    'question': qa.question,
                    'expected_answer': qa.answer,
                    'generated_answer': result['answer'],
                    'context': context,
                    'judge_results': judge_results,
                    'query_time': query_time,
                    'retrieval_time': result.get('retrieval_time', 0.0)
                }
                
                qa_results.append(qa_result)
                
            except Exception as e:
                qa_result = {
                    'question': qa.question,
                    'expected_answer': qa.answer,
                    'generated_answer': f"Error: {str(e)}",
                    'context': "",
                    'judge_results': {
                        'correctness': {'score': 1, 'rationale': f"Query failed: {str(e)}"},
                        'relevance': {'score': 1, 'rationale': f"Query failed: {str(e)}"},
                        'groundedness': {'score': 1, 'rationale': f"Query failed: {str(e)}"}
                    },
                    'query_time': 0.0,
                    'retrieval_time': 0.0
                }
                qa_results.append(qa_result)
        
        total_time = time.time() - start_time
        avg_scores = self._calculate_average_scores(qa_results)
        
        return BenchmarkResult(
            config=config,
            qa_results=qa_results,
            avg_scores=avg_scores,
            total_queries=len(qa_dataset),
            avg_execution_time=total_time / len(qa_dataset),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def run_grid_search(self, qa_dataset: list[SyntheticQA], 
                       chunking_strategies: list[str] = None,
                       chunk_sizes: list[int] = None,
                       embedding_models: list[str] = None,
                       retrieval_strategies: list[str] = None,
                       top_k_values: list[int] = None,
                       languages: list[str] = None) -> list[BenchmarkResult]:
        
        chunking_strategies = chunking_strategies or ["markdown_header", "sentence"]
        chunk_sizes = chunk_sizes or [500, 1000]
        embedding_models = embedding_models or ["sentence-transformers/all-MiniLM-L6-v2"]
        retrieval_strategies = retrieval_strategies or ["semantic", "hybrid"]
        top_k_values = top_k_values or [3, 5]
        languages = languages or ["english"]
        
        configs = []
        for chunking, chunk_size, embedding, retrieval, top_k, language in product(
            chunking_strategies, chunk_sizes, embedding_models, 
            retrieval_strategies, top_k_values, languages
        ):
            config = BenchmarkConfig(
                chunking_strategy=chunking,
                chunk_size=chunk_size,
                embedding_model=embedding,
                retrieval_strategy=retrieval,
                top_k=top_k,
                language=language
            )
            configs.append(config)
        
        results = []
        for i, config in enumerate(configs):
            print(f"Running benchmark {i+1}/{len(configs)}: {config.get_identifier()}")
            try:
                result = self.run_single_config(config, qa_dataset)
                results.append(result)
            except Exception as e:
                print(f"Error running config {config.get_identifier()}: {e}")
                continue
        
        return results
    
    def _apply_config_to_settings(self, config: BenchmarkConfig) -> Settings:
        modified_settings = Settings()
        modified_settings.chunking_strategy = config.chunking_strategy
        modified_settings.chunk_size = config.chunk_size
        modified_settings.embedding_model = config.embedding_model
        modified_settings.retrieval_strategy = config.retrieval_strategy
        modified_settings.top_k = config.top_k
        return modified_settings
    
    def _extract_context_from_result(self, result: dict[str, Any]) -> str:
        if 'context' in result:
            return result['context']
        elif 'documents' in result:
            return "\n".join([doc.get('content', '') for doc in result['documents']])
        return ""
    
    def _calculate_average_scores(self, qa_results: list[dict[str, Any]]) -> dict[str, float]:
        if not qa_results:
            return {}
        
        score_sums = {}
        for qa_result in qa_results:
            for criteria, judge_result in qa_result.get('judge_results', {}).items():
                if criteria not in score_sums:
                    score_sums[criteria] = []
                score_sums[criteria].append(judge_result['score'])
        
        avg_scores = {}
        for criteria, scores in score_sums.items():
            avg_scores[criteria] = sum(scores) / len(scores)
        
        if avg_scores:
            avg_scores['overall'] = sum(avg_scores.values()) / len(avg_scores)
        
        return avg_scores
    
    def save_results(self, results: list[BenchmarkResult], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            filename = f"benchmark_{result.config.get_identifier()}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        summary_data = []
        for result in results:
            summary_data.append({
                'config': result.config.to_dict(),
                'avg_scores': result.avg_scores,
                'total_queries': result.total_queries,
                'avg_execution_time': result.avg_execution_time,
                'timestamp': result.timestamp
            })
        
        summary_filepath = output_path / "benchmark_summary.json"
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results)} benchmark results to {output_path}")
    
    def load_results(self, output_dir: str) -> list[BenchmarkResult]:
        output_path = Path(output_dir)
        results = []
        
        for filepath in output_path.glob("benchmark_*.json"):
            if filepath.name == "benchmark_summary.json":
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            config = BenchmarkConfig(**data['config'])
            result = BenchmarkResult(
                config=config,
                qa_results=data['qa_results'],
                avg_scores=data['avg_scores'],
                total_queries=data['total_queries'],
                avg_execution_time=data['avg_execution_time'],
                timestamp=data['timestamp']
            )
            results.append(result)
        
        return results
