#!/usr/bin/env python3

import argparse
from pathlib import Path

from config.settings import Settings
from src.evaluation.synthetic_dataset import SyntheticDatasetGenerator
from src.evaluation.benchmark import RAGBenchmark
from src.evaluation.reporter import BenchmarkReporter
from src.evaluation.comparative_benchmark import ComparativeBenchmark, ComparativeConfig
from src.evaluation.comparative_reporter import ComparativeReporter
from src.storage.database import DatabaseManager


def generate_synthetic_dataset(settings: Settings, n_samples: int = 30) -> str:
    print(f"Generating synthetic evaluation dataset with {n_samples} samples...")
    
    generator = SyntheticDatasetGenerator(settings)
    database_manager = DatabaseManager(settings)
    
    with database_manager as db:
        qa_pairs, generation_stats = generator.generate_dataset_with_stats(db, n_samples)
    
    if not qa_pairs:
        raise ValueError("Failed to generate any QA pairs")
    
    filtered_pairs = generator.filter_dataset(qa_pairs)
    
    if not filtered_pairs:
        print("Warning: No QA pairs passed filtering criteria. Lowering standards...")
        filtered_pairs = generator.filter_dataset(qa_pairs, min_groundedness=3, min_relevance=3, min_standalone=3)
    
    dataset_path = "data/evaluation/synthetic_dataset.json"
    generator.save_dataset(filtered_pairs, dataset_path)
    
    print(f"Generated {len(filtered_pairs)} high-quality QA pairs")
    
    print("\n" + "="*60)
    print("DATASET GENERATION REPORT")
    print("="*60)
    print(f"Total documents processed: {generation_stats['total_documents']}")
    print(f"Total QA pairs generated: {len(qa_pairs)}")
    print(f"QA pairs after filtering: {len(filtered_pairs)}")
    print(f"Filter success rate: {len(filtered_pairs)/len(qa_pairs)*100:.1f}%")
    
    print("\nGeneration by source document:")
    for doc_id, count in generation_stats['by_document'].items():
        print(f"  {doc_id}: {count} pairs")
    
    print(f"\nAverage scores:")
    if generation_stats['avg_scores']:
        for metric, score in generation_stats['avg_scores'].items():
            print(f"  {metric}: {score:.2f}/5.0")
    
    return dataset_path


def run_comparative_study(settings: Settings, dataset_path: str, quick: bool = False) -> str:
    print("Loading synthetic dataset...")
    generator = SyntheticDatasetGenerator(settings)
    qa_dataset = generator.load_dataset(dataset_path)
    
    if not qa_dataset:
        raise ValueError(f"No QA pairs found in {dataset_path}")
    
    print(f"Running comparative study on {len(qa_dataset)} QA pairs...")
    
    configs = [
        ComparativeConfig("baseline"),
        ComparativeConfig("full_context"),
        ComparativeConfig("rag", {"top_k": 3}),
        ComparativeConfig("rag", {"top_k": 5})
    ]
    
    if not quick:
        configs.extend([
            ComparativeConfig("rag", {"top_k": 10}),
            ComparativeConfig("rag", {"top_k": 15})
        ])
    
    benchmark = ComparativeBenchmark(settings)
    results = benchmark.run_comparative_study(qa_dataset, configs)
    
    if not results:
        raise ValueError("No comparative results generated")
    
    output_dir = "data/evaluation/comparative_results"
    benchmark.save_results(results, output_dir)
    
    print(f"Comparative study complete! Generated {len(results)} variant results")
    return output_dir


def generate_comparative_report(results_dir: str):
    print("Generating comparative evaluation report...")
    
    benchmark = ComparativeBenchmark(Settings())
    results = benchmark.load_results(results_dir)
    
    if not results:
        raise ValueError(f"No comparative results found in {results_dir}")
    
    reporter = ComparativeReporter()
    reporter.add_results(results)
    
    report_dir = "data/evaluation/comparative_reports"
    reporter.save_report(report_dir)
    reporter.print_summary()
    
    print(f"Comparative reports saved to {report_dir}")


def run_benchmark(settings: Settings, dataset_path: str, quick: bool = False) -> str:
    print("Loading synthetic dataset...")
    generator = SyntheticDatasetGenerator(settings)
    qa_dataset = generator.load_dataset(dataset_path)
    
    if not qa_dataset:
        raise ValueError(f"No QA pairs found in {dataset_path}")
    
    print(f"Running benchmark on {len(qa_dataset)} QA pairs...")
    
    benchmark = RAGBenchmark(settings)
    
    if quick:
        chunking_strategies = ["markdown_header"]
        chunk_sizes = [500, 1000]
        retrieval_strategies = ["semantic"]
        top_k_values = [3, 5]
    else:
        chunking_strategies = ["markdown_header", "sentence", "fixed_size"]
        chunk_sizes = [200, 500, 1000, 2000]
        retrieval_strategies = ["semantic", "hybrid"]
        top_k_values = [3, 5, 10]
    
    results = benchmark.run_grid_search(
        qa_dataset=qa_dataset,
        chunking_strategies=chunking_strategies,
        chunk_sizes=chunk_sizes,
        retrieval_strategies=retrieval_strategies,
        top_k_values=top_k_values
    )
    
    if not results:
        raise ValueError("No benchmark results generated")
    
    output_dir = "data/evaluation/benchmark_results"
    benchmark.save_results(results, output_dir)
    
    print(f"Benchmark complete! Generated {len(results)} configuration results")
    return output_dir


def generate_report(results_dir: str):
    print("Generating evaluation report...")
    
    benchmark = RAGBenchmark(Settings())
    results = benchmark.load_results(results_dir)
    
    if not results:
        raise ValueError(f"No benchmark results found in {results_dir}")
    
    reporter = BenchmarkReporter()
    reporter.add_results(results)
    
    report_dir = "data/evaluation/reports"
    reporter.save_report(report_dir)
    reporter.print_summary()
    
    print(f"Reports saved to {report_dir}")


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation System")
    parser.add_argument("command", choices=["generate", "benchmark", "report", "compare", "comparative-report", "full"], 
                       help="Command to run")
    parser.add_argument("--samples", type=int, default=30, 
                       help="Number of synthetic QA pairs to generate")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick benchmark with fewer configurations")
    parser.add_argument("--test", action="store_true", 
                       help="Test mode: generate only 2 samples and use minimal configs")
    parser.add_argument("--dataset", type=str, default="data/evaluation/synthetic_dataset.json",
                       help="Path to synthetic dataset")
    parser.add_argument("--results", type=str, default="data/evaluation/benchmark_results",
                       help="Path to benchmark results directory")
    parser.add_argument("--comparative-results", type=str, default="data/evaluation/comparative_results",
                       help="Path to comparative results directory")
    
    args = parser.parse_args()
    
    settings = Settings()
    
    try:
        if args.command == "generate":
            samples = 2 if args.test else args.samples
            dataset_path = generate_synthetic_dataset(settings, samples)
            print(f"Synthetic dataset generated: {dataset_path}")
        
        elif args.command == "benchmark":
            if not Path(args.dataset).exists():
                print(f"Dataset not found: {args.dataset}")
                print("Run 'python scripts/run_evaluation.py generate' first")
                return
            
            quick = args.quick or args.test
            results_dir = run_benchmark(settings, args.dataset, quick)
            print(f"Benchmark results saved: {results_dir}")
        
        elif args.command == "compare":
            if not Path(args.dataset).exists():
                print(f"Dataset not found: {args.dataset}")
                print("Run 'python scripts/run_evaluation.py generate' first")
                return
            
            quick = args.quick or args.test
            results_dir = run_comparative_study(settings, args.dataset, quick)
            print(f"Comparative results saved: {results_dir}")
        
        elif args.command == "comparative-report":
            if not Path(args.comparative_results).exists():
                print(f"Comparative results directory not found: {args.comparative_results}")
                print("Run comparative study first")
                return
            
            generate_comparative_report(args.comparative_results)
        
        elif args.command == "report":
            if not Path(args.results).exists():
                print(f"Results directory not found: {args.results}")
                print("Run benchmark first")
                return
            
            generate_report(args.results)
        
        elif args.command == "full":
            print("Running full evaluation pipeline...")
            
            samples = 2 if args.test else args.samples
            quick = args.quick or args.test
            dataset_path = generate_synthetic_dataset(settings, samples)
            results_dir = run_benchmark(settings, dataset_path, quick)
            generate_report(results_dir)
            
            print("Running comparative study...")
            comp_results_dir = run_comparative_study(settings, dataset_path, quick)
            generate_comparative_report(comp_results_dir)
            
            print("\nFull evaluation complete!")
            print(f"Dataset: {dataset_path}")
            print(f"Results: {results_dir}")
            print(f"Comparative Results: {comp_results_dir}")
            print(f"Reports: data/evaluation/reports")
            print(f"Comparative Reports: data/evaluation/comparative_reports")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
