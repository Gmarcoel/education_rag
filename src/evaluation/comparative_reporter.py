from typing import Any
import json
from pathlib import Path
from dataclasses import dataclass

from src.evaluation.comparative_benchmark import ComparativeResult


@dataclass
class ComparisonInsight:
    category: str
    message: str
    impact: str
    recommendation: str


class ComparativeReporter:
    
    def __init__(self):
        self.results: list[ComparativeResult] = []
    
    def add_results(self, results: list[ComparativeResult]):
        self.results.extend(results)
    
    def generate_comparative_report(self) -> dict[str, Any]:
        if not self.results:
            return {"error": "No comparative results available"}
        
        variant_analysis = self._analyze_variants()
        performance_analysis = self._analyze_performance()
        trade_off_analysis = self._analyze_trade_offs()
        insights = self._generate_insights()
        
        return {
            "summary": {
                "total_variants": len(self.results),
                "variants_tested": [r.config.get_identifier() for r in self.results]
            },
            "variant_analysis": variant_analysis,
            "performance_analysis": performance_analysis,
            "trade_off_analysis": trade_off_analysis,
            "insights": [insight.__dict__ for insight in insights],
            "detailed_comparison": self._create_detailed_comparison()
        }
    
    def _analyze_variants(self) -> dict[str, Any]:
        variant_scores = {}
        variant_performance = {}
        
        for result in self.results:
            variant_id = result.config.get_identifier()
            variant_scores[variant_id] = result.avg_scores
            variant_performance[variant_id] = result.avg_performance
        
        best_accuracy = max(self.results, key=lambda r: r.avg_scores.get('overall', 0))
        fastest_variant = min(self.results, key=lambda r: r.avg_performance.get('response_time', float('inf')))
        lowest_tokens = min(self.results, key=lambda r: r.avg_performance.get('input_tokens', float('inf')))
        
        return {
            "best_accuracy": {
                "variant": best_accuracy.config.get_identifier(),
                "score": round(best_accuracy.avg_scores.get('overall', 0), 3),
                "details": best_accuracy.avg_scores
            },
            "fastest": {
                "variant": fastest_variant.config.get_identifier(),
                "response_time": round(fastest_variant.avg_performance.get('response_time', 0), 3),
                "accuracy": round(fastest_variant.avg_scores.get('overall', 0), 3)
            },
            "lowest_tokens": {
                "variant": lowest_tokens.config.get_identifier(),
                "input_tokens": round(lowest_tokens.avg_performance.get('input_tokens', 0), 3),
                "accuracy": round(lowest_tokens.avg_scores.get('overall', 0), 3)
            }
        }
    
    def _analyze_performance(self) -> dict[str, Any]:
        performance_metrics = {}
        
        for result in self.results:
            variant_id = result.config.get_identifier()
            performance_metrics[variant_id] = {
                "response_time": result.avg_performance.get('response_time', 0),
                "input_tokens": result.avg_performance.get('input_tokens', 0),
                "output_tokens": result.avg_performance.get('output_tokens', 0),
                "context_size": result.avg_performance.get('context_size', 0),
                "api_calls": result.avg_performance.get('api_calls', 0)
            }
        
        return performance_metrics
    
    def _analyze_trade_offs(self) -> dict[str, Any]:
        trade_offs = []
        
        for result in self.results:
            accuracy = result.avg_scores.get('overall', 0)
            speed = 1 / max(result.avg_performance.get('response_time', 0.001), 0.001)
            efficiency = 1 / max(result.avg_performance.get('input_tokens', 1), 1)
            
            trade_offs.append({
                "variant": result.config.get_identifier(),
                "accuracy": round(accuracy, 3),
                "speed_score": round(speed, 3),
                "efficiency_score": round(efficiency, 3),
                "balanced_score": round((accuracy + speed + efficiency) / 3, 3)
            })
        
        if not trade_offs:
            return {"error": "No trade-off data available"}
        
        best_balanced = max(trade_offs, key=lambda x: x['balanced_score'])
        
        return {
            "all_variants": trade_offs,
            "best_balanced": best_balanced,
            "recommendations": self._generate_use_case_recommendations()
        }
    
    def _generate_use_case_recommendations(self) -> dict[str, str]:
        if len(self.results) < 2:
            return {}
        
        fastest = min(self.results, key=lambda r: r.avg_performance.get('response_time', float('inf')))
        most_accurate = max(self.results, key=lambda r: r.avg_scores.get('overall', 0))
        lowest_tokens = min(self.results, key=lambda r: r.avg_performance.get('input_tokens', float('inf')))
        
        return {
            "real_time_applications": fastest.config.get_identifier(),
            "high_accuracy_requirements": most_accurate.config.get_identifier(),
            "resource_constrained_environments": lowest_tokens.config.get_identifier()
        }
    
    def _generate_insights(self) -> list[ComparisonInsight]:
        insights = []
        
        if len(self.results) >= 2:
            speed_range = self._get_metric_range('response_time')
            accuracy_range = self._get_metric_range('overall', is_score=True)
            
            if speed_range['max'] / speed_range['min'] > 2:
                insights.append(ComparisonInsight(
                    category="Performance",
                    message=f"Response time varies significantly ({speed_range['min']:.2f}s to {speed_range['max']:.2f}s)",
                    impact="High",
                    recommendation="Choose variant based on latency requirements"
                ))
            
            if accuracy_range['max'] - accuracy_range['min'] > 1.0:
                insights.append(ComparisonInsight(
                    category="Accuracy",
                    message=f"Accuracy scores show large variance ({accuracy_range['min']:.2f} to {accuracy_range['max']:.2f})",
                    impact="High",
                    recommendation="Prioritize accuracy for critical applications"
                ))
        
        return insights
    
    def _get_metric_range(self, metric: str, is_score: bool = False) -> dict[str, float]:
        if is_score:
            values = [r.avg_scores.get(metric, 0) for r in self.results]
        else:
            values = [r.avg_performance.get(metric, 0) for r in self.results]
        
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values)
        }
    
    def _create_detailed_comparison(self) -> list[dict[str, Any]]:
        comparison = []
        
        for result in self.results:
            comparison.append({
                "variant": result.config.get_identifier(),
                "config": result.config.to_dict(),
                "scores": result.avg_scores,
                "performance": result.avg_performance,
                "total_queries": result.total_queries
            })
        
        return sorted(comparison, key=lambda x: x['scores'].get('overall', 0), reverse=True)
    
    def save_report(self, report_path: str):
        report_dir = Path(report_path)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        comparative_report = self.generate_comparative_report()
        
        with open(report_dir / "comparative_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(comparative_report, f, indent=2, ensure_ascii=False)
        
        print(f"Comparative reports saved to {report_dir}")
    
    def print_summary(self):
        if not self.results:
            print("No comparative results available")
            return
        
        report = self.generate_comparative_report()
        
        print("\n" + "="*70)
        print("COMPARATIVE EVALUATION SUMMARY")
        print("="*70)
        
        print(f"Total Variants Tested: {report['summary']['total_variants']}")
        print("\nVariants:")
        for variant in report['summary']['variants_tested']:
            print(f"  • {variant}")
        
        print("\nBest Performers:")
        
        best_accuracy = report['variant_analysis']['best_accuracy']
        print(f"\nMost Accurate:")
        print(f"  Variant: {best_accuracy['variant']}")
        print(f"  Overall Score: {best_accuracy['score']:.3f}/5.0")
        for metric, score in best_accuracy['details'].items():
            if metric != 'overall':
                print(f"  {metric.title()}: {score:.3f}/5.0")
        
        fastest = report['variant_analysis']['fastest']
        print(f"\nFastest:")
        print(f"  Variant: {fastest['variant']}")
        print(f"  Response Time: {fastest['response_time']:.3f}s")
        print(f"  Accuracy: {fastest['accuracy']:.3f}/5.0")
        
        lowest_tokens = report['variant_analysis']['lowest_tokens']
        print(f"\nLowest Token Usage:")
        print(f"  Variant: {lowest_tokens['variant']}")
        print(f"  Input Tokens: {lowest_tokens['input_tokens']:.0f}")
        print(f"  Accuracy: {lowest_tokens['accuracy']:.3f}/5.0")
        
        print(f"\nRecommendations:")
        recommendations = report['trade_off_analysis']['recommendations']
        for use_case, variant in recommendations.items():
            print(f"  {use_case.replace('_', ' ').title()}: {variant}")
        
        print("\nKey Insights:")
        for insight in report['insights']:
            print(f"  • {insight['message']}")
            print(f"    Impact: {insight['impact']} | {insight['recommendation']}")
        
        print("\n" + "="*70)
