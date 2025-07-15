from dataclasses import dataclass, field
from typing import Any, Optional
import time
import json
from pathlib import Path
from itertools import product

from config.settings import Settings
from src.evaluation.judge import JudgeFactory, JudgeResult
from src.evaluation.synthetic_dataset import SyntheticQA
from src.evaluation.system_variants import SystemVariant, SystemVariantFactory, PerformanceMetrics


@dataclass
class ComparativeConfig:
    variant_type: str
    variant_params: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_type": self.variant_type,
            "variant_params": self.variant_params
        }
    
    def get_identifier(self) -> str:
        params_str = "_".join([f"{k}_{v}" for k, v in self.variant_params.items()])
        return f"{self.variant_type}_{params_str}" if params_str else self.variant_type


@dataclass
class ComparativeResult:
    config: ComparativeConfig
    qa_results: list[dict[str, Any]] = field(default_factory=list)
    avg_scores: dict[str, float] = field(default_factory=dict)
    avg_performance: dict[str, float] = field(default_factory=dict)
    total_queries: int = 0
    timestamp: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "qa_results": self.qa_results,
            "avg_scores": self.avg_scores,
            "avg_performance": self.avg_performance,
            "total_queries": self.total_queries,
            "timestamp": self.timestamp
        }


class ComparativeBenchmark:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.judges = [
            JudgeFactory.create_judge("correctness", settings),
            JudgeFactory.create_judge("relevance", settings),
            JudgeFactory.create_judge("groundedness", settings)
        ]
    
    def run_single_variant(self, qa_dataset: list[SyntheticQA], config: ComparativeConfig) -> ComparativeResult:
        print(f"Running variant: {config.get_identifier()}")
        
        variant = SystemVariantFactory.create_variant(
            config.variant_type,
            self.settings,
            **config.variant_params
        )
        
        start_time = time.time()
        qa_results = []
        
        for qa in qa_dataset:
            print(f"Processing query: {qa.question[:50]}...")
            
            try:
                answer_result = variant.answer_question(qa.question)
                
                if answer_result.error:
                    qa_result = self._create_error_result(qa, answer_result.error)
                else:
                    judge_results = self._evaluate_answer(qa, answer_result)
                    qa_result = {
                        'question': qa.question,
                        'expected_answer': qa.answer,
                        'generated_answer': answer_result.answer,
                        'context': answer_result.context,
                        'judge_results': judge_results,
                        'performance_metrics': {
                            'response_time': answer_result.performance.response_time,
                            'input_tokens': answer_result.performance.input_tokens,
                            'output_tokens': answer_result.performance.output_tokens,
                            'api_calls': answer_result.performance.api_calls,
                            'context_size': answer_result.performance.context_size
                        }
                    }
                
                qa_results.append(qa_result)
                
            except Exception as e:
                qa_result = self._create_error_result(qa, str(e))
                qa_results.append(qa_result)
        
        total_time = time.time() - start_time
        avg_scores = self._calculate_average_scores(qa_results)
        avg_performance = self._calculate_average_performance(qa_results)
        
        return ComparativeResult(
            config=config,
            qa_results=qa_results,
            avg_scores=avg_scores,
            avg_performance=avg_performance,
            total_queries=len(qa_dataset),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def run_comparative_study(self, qa_dataset: list[SyntheticQA], 
                             variant_configs: list[ComparativeConfig]) -> list[ComparativeResult]:
        results = []
        
        for config in variant_configs:
            try:
                result = self.run_single_variant(qa_dataset, config)
                results.append(result)
                print(f"Completed variant: {config.get_identifier()}")
            except Exception as e:
                print(f"Error running variant {config.get_identifier()}: {e}")
                continue
        
        return results
    
    def _evaluate_answer(self, qa: SyntheticQA, answer_result) -> dict[str, dict[str, Any]]:
        judge_results = {}
        
        for judge in self.judges:
            try:
                judge_result = judge.evaluate(
                    qa.question, 
                    answer_result.answer, 
                    qa.answer, 
                    answer_result.context
                )
                judge_results[judge_result.criteria] = {
                    'score': judge_result.score,
                    'rationale': judge_result.rationale
                }
            except Exception as e:
                judge_results[judge.criteria] = {
                    'score': 1,
                    'rationale': f"Judge evaluation failed: {str(e)}"
                }
        
        return judge_results
    
    def _create_error_result(self, qa: SyntheticQA, error: str) -> dict[str, Any]:
        return {
            'question': qa.question,
            'expected_answer': qa.answer,
            'generated_answer': f"Error: {error}",
            'context': "",
            'judge_results': {
                'correctness': {'score': 1, 'rationale': f"Query failed: {error}"},
                'relevance': {'score': 1, 'rationale': f"Query failed: {error}"},
                'groundedness': {'score': 1, 'rationale': f"Query failed: {error}"}
            },
            'performance_metrics': {
                'response_time': 0.0,
                'input_tokens': 0,
                'output_tokens': 0,
                'api_calls': 0,
                'context_size': 0
            }
        }
    
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
    
    def _calculate_average_performance(self, qa_results: list[dict[str, Any]]) -> dict[str, float]:
        if not qa_results:
            return {}
        
        metric_sums = {}
        for qa_result in qa_results:
            for metric, value in qa_result.get('performance_metrics', {}).items():
                if metric not in metric_sums:
                    metric_sums[metric] = []
                metric_sums[metric].append(value)
        
        avg_performance = {}
        for metric, values in metric_sums.items():
            avg_performance[metric] = sum(values) / len(values)
        
        return avg_performance
    
    def save_results(self, results: list[ComparativeResult], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            filename = f"comparative_{result.config.get_identifier()}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        summary_data = []
        for result in results:
            summary_data.append({
                'config': result.config.to_dict(),
                'avg_scores': result.avg_scores,
                'avg_performance': result.avg_performance,
                'total_queries': result.total_queries,
                'timestamp': result.timestamp
            })
        
        summary_filepath = output_path / "comparative_summary.json"
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    def load_results(self, results_dir: str) -> list[ComparativeResult]:
        results_path = Path(results_dir)
        if not results_path.exists():
            return []
        
        results = []
        for filepath in results_path.glob("comparative_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                config = ComparativeConfig(
                    variant_type=data['config']['variant_type'],
                    variant_params=data['config']['variant_params']
                )
                
                result = ComparativeResult(
                    config=config,
                    qa_results=data['qa_results'],
                    avg_scores=data['avg_scores'],
                    avg_performance=data['avg_performance'],
                    total_queries=data['total_queries'],
                    timestamp=data['timestamp']
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error loading result from {filepath}: {e}")
                continue
        
        return results
