from typing import Any
import json
from pathlib import Path
from dataclasses import dataclass

from src.evaluation.benchmark import BenchmarkResult


@dataclass
class PerformanceInsight:
    category: str
    message: str
    impact: str
    recommendation: str


class BenchmarkReporter:
    
    def __init__(self):
        self.results: list[BenchmarkResult] = []
    
    def add_results(self, results: list[BenchmarkResult]):
        self.results.extend(results)
    
    def generate_summary_report(self) -> dict[str, Any]:
        if not self.results:
            return {"error": "No benchmark results available"}
        
        best_config = max(self.results, key=lambda r: r.avg_scores.get('overall', 0))
        worst_config = min(self.results, key=lambda r: r.avg_scores.get('overall', 0))
        
        avg_performance = sum(r.avg_scores.get('overall', 0) for r in self.results) / len(self.results)
        
        config_analysis = self._analyze_config_impact()
        performance_insights = self._generate_insights()
        
        return {
            "summary": {
                "total_configurations": len(self.results),
                "average_performance": round(avg_performance, 3),
                "best_config": {
                    "config": best_config.config.to_dict(),
                    "score": round(best_config.avg_scores.get('overall', 0), 3),
                    "execution_time": round(best_config.avg_execution_time, 3)
                },
                "worst_config": {
                    "config": worst_config.config.to_dict(),
                    "score": round(worst_config.avg_scores.get('overall', 0), 3),
                    "execution_time": round(worst_config.avg_execution_time, 3)
                }
            },
            "config_analysis": config_analysis,
            "insights": [insight.__dict__ for insight in performance_insights],
            "detailed_results": [
                {
                    "config": r.config.to_dict(),
                    "scores": r.avg_scores,
                    "execution_time": r.avg_execution_time,
                    "total_queries": r.total_queries
                } for r in self.results
            ]
        }
    
    def generate_comparison_report(self, baseline_config: str) -> dict[str, Any]:
        baseline_result = next((r for r in self.results if r.config.get_identifier() == baseline_config), None)
        
        if not baseline_result:
            return {"error": f"Baseline config {baseline_config} not found"}
        
        comparisons = []
        for result in self.results:
            if result.config.get_identifier() == baseline_config:
                continue
            
            comparison = {
                "config": result.config.to_dict(),
                "score_diff": result.avg_scores.get('overall', 0) - baseline_result.avg_scores.get('overall', 0),
                "time_diff": result.avg_execution_time - baseline_result.avg_execution_time,
                "improvement": result.avg_scores.get('overall', 0) > baseline_result.avg_scores.get('overall', 0)
            }
            comparisons.append(comparison)
        
        comparisons.sort(key=lambda x: x['score_diff'], reverse=True)
        
        return {
            "baseline": {
                "config": baseline_result.config.to_dict(),
                "score": baseline_result.avg_scores.get('overall', 0),
                "execution_time": baseline_result.avg_execution_time
            },
            "comparisons": comparisons,
            "improvements": [c for c in comparisons if c['improvement']],
            "regressions": [c for c in comparisons if not c['improvement']]
        }
    
    def generate_detailed_analysis(self) -> dict[str, Any]:
        if not self.results:
            return {"error": "No benchmark results available"}
        
        score_distribution = self._analyze_score_distribution()
        failure_analysis = self._analyze_failures()
        performance_trends = self._analyze_performance_trends()
        
        return {
            "score_distribution": score_distribution,
            "failure_analysis": failure_analysis,
            "performance_trends": performance_trends,
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_config_impact(self) -> dict[str, Any]:
        impact_analysis = {}
        
        for param in ['chunking_strategy', 'chunk_size', 'retrieval_strategy', 'top_k']:
            param_groups = {}
            for result in self.results:
                param_value = getattr(result.config, param)
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result.avg_scores.get('overall', 0))
            
            param_analysis = {}
            for value, scores in param_groups.items():
                param_analysis[str(value)] = {
                    "avg_score": round(sum(scores) / len(scores), 3),
                    "count": len(scores)
                }
            
            impact_analysis[param] = param_analysis
        
        return impact_analysis
    
    def _generate_insights(self) -> list[PerformanceInsight]:
        insights = []
        
        if len(self.results) < 2:
            return insights
        
        chunk_sizes = {}
        for result in self.results:
            size = result.config.chunk_size
            if size not in chunk_sizes:
                chunk_sizes[size] = []
            chunk_sizes[size].append(result.avg_scores.get('overall', 0))
        
        if len(chunk_sizes) > 1:
            best_size = max(chunk_sizes.items(), key=lambda x: sum(x[1]) / len(x[1]))
            worst_size = min(chunk_sizes.items(), key=lambda x: sum(x[1]) / len(x[1]))
            
            if best_size[0] != worst_size[0]:
                insights.append(PerformanceInsight(
                    category="chunking",
                    message=f"Chunk size {best_size[0]} performs {((sum(best_size[1]) / len(best_size[1])) - (sum(worst_size[1]) / len(worst_size[1]))):.2f} points better than {worst_size[0]}",
                    impact="high",
                    recommendation=f"Consider using chunk size {best_size[0]} for better performance"
                ))
        
        retrieval_strategies = {}
        for result in self.results:
            strategy = result.config.retrieval_strategy
            if strategy not in retrieval_strategies:
                retrieval_strategies[strategy] = []
            retrieval_strategies[strategy].append(result.avg_scores.get('overall', 0))
        
        if len(retrieval_strategies) > 1:
            best_strategy = max(retrieval_strategies.items(), key=lambda x: sum(x[1]) / len(x[1]))
            insights.append(PerformanceInsight(
                category="retrieval",
                message=f"Retrieval strategy '{best_strategy[0]}' shows superior performance",
                impact="medium",
                recommendation=f"Use {best_strategy[0]} retrieval for optimal results"
            ))
        
        return insights
    
    def _analyze_score_distribution(self) -> dict[str, Any]:
        all_scores = [r.avg_scores.get('overall', 0) for r in self.results]
        
        return {
            "min": min(all_scores),
            "max": max(all_scores),
            "mean": sum(all_scores) / len(all_scores),
            "median": sorted(all_scores)[len(all_scores) // 2],
            "std_dev": self._calculate_std_dev(all_scores)
        }
    
    def _analyze_failures(self) -> dict[str, Any]:
        failure_patterns = {}
        
        for result in self.results:
            for qa_result in result.qa_results:
                if qa_result['generated_answer'].startswith('Error:'):
                    error_type = qa_result['generated_answer'].split(':')[1].strip()
                    if error_type not in failure_patterns:
                        failure_patterns[error_type] = 0
                    failure_patterns[error_type] += 1
        
        return {
            "total_failures": sum(failure_patterns.values()),
            "failure_types": failure_patterns
        }
    
    def _analyze_performance_trends(self) -> dict[str, Any]:
        execution_times = [r.avg_execution_time for r in self.results]
        
        return {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "fastest_config": min(self.results, key=lambda r: r.avg_execution_time).config.to_dict(),
            "slowest_config": max(self.results, key=lambda r: r.avg_execution_time).config.to_dict()
        }
    
    def _generate_recommendations(self) -> list[str]:
        recommendations = []
        
        best_result = max(self.results, key=lambda r: r.avg_scores.get('overall', 0))
        
        recommendations.append(f"Use chunking strategy '{best_result.config.chunking_strategy}' for best performance")
        recommendations.append(f"Set chunk size to {best_result.config.chunk_size} tokens")
        recommendations.append(f"Use '{best_result.config.retrieval_strategy}' retrieval with top_k={best_result.config.top_k}")
        
        if best_result.avg_scores.get('overall', 0) < 3.0:
            recommendations.append("Consider tuning prompt templates or using different embedding models")
        
        return recommendations
    
    def _calculate_std_dev(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def save_report(self, output_path: str):
        report_path = Path(output_path)
        report_path.mkdir(parents=True, exist_ok=True)
        
        summary_report = self.generate_summary_report()
        with open(report_path / "summary_report.json", 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        detailed_analysis = self.generate_detailed_analysis()
        with open(report_path / "detailed_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"Reports saved to {report_path}")
    
    def print_summary(self):
        if not self.results:
            print("No benchmark results available")
            return
        
        summary = self.generate_summary_report()
        
        print("\n" + "="*60)
        print("RAG BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Total Configurations Tested: {summary['summary']['total_configurations']}")
        print(f"Average Performance: {summary['summary']['average_performance']:.3f}/5.0")
        
        print("\nBest Configuration:")
        best = summary['summary']['best_config']
        print(f"  Score: {best['score']:.3f}/5.0")
        print(f"  Execution Time: {best['execution_time']:.3f}s")
        for key, value in best['config'].items():
            print(f"  {key}: {value}")
        
        print("\nKey Insights:")
        for insight in summary['insights']:
            print(f"  • {insight['message']}")
            print(f"    Recommendation: {insight['recommendation']}")
        
        print("\n" + "="*60)
