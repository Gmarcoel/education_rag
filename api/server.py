import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.settings import Settings
from scripts.demo import RAGPipeline
from src.ingestion.loader import FileSystemLoader
from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.chunker import ChunkerFactory
from src.embedding.encoder import DocumentEncoder
from src.generation.translator import TranslationService
from src.evaluation.synthetic_dataset import SyntheticDatasetGenerator
from src.evaluation.benchmark import RAGBenchmark, BenchmarkConfig
from src.evaluation.reporter import BenchmarkReporter


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    template: str | None = 'BASIC_QA'


class QueryResponse(BaseModel):
    query: str
    answer: str
    total_documents: int
    retrieval_time: float


class QuerySpanishResponse(BaseModel):
    query: str
    original_query: str
    english_query: str
    answer: str
    english_answer: str
    total_documents: int
    retrieval_time: float
    translation_time: float


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content: str
    metadata: dict[str, object]


class ChunkResponse(BaseModel):
    id: str
    content: str
    metadata: dict[str, object]
    parent_document_id: str


class EmbeddingResponse(BaseModel):
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, object]


class RetrievalResponse(BaseModel):
    query: str
    documents: list[dict[str, object]]
    execution_time: float
    total_results: int


class IndexingStepResponse(BaseModel):
    step: str
    status: str
    data: dict[str, object]
    execution_time: float


class ChunkingRequest(BaseModel):
    document_id: str
    strategy: str = 'markdown_header'


class EmbeddingRequest(BaseModel):
    texts: list[str]


from contextlib import asynccontextmanager

settings = Settings()
pipeline = RAGPipeline(settings)
translation_service = TranslationService(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    stats = pipeline.get_database_stats()
    if stats["document_count"] == 0:
        print("Building index on startup...")
        result = pipeline.build_index()
        print(f"Index built: {result}")
    else:
        print(f"Using existing index with {stats['document_count']} documents")
    yield

app = FastAPI(title="RAG Educational System", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "RAG Educational System API", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = pipeline.query(request.query, request.top_k, request.template)
        return QueryResponse(
            query=result['query'],
            answer=result['answer'],
            total_documents=result['total_documents'],
            retrieval_time=result['retrieval_time']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-spanish", response_model=QuerySpanishResponse)
async def query_spanish(request: QueryRequest):
    try:
        import time
        
        translation_service = TranslationService()
        original_query = request.query
        
        translation_start = time.time()
        english_query = translation_service.process_multilingual_query(original_query)
        
        english_request = QueryRequest(
            query=english_query,
            top_k=request.top_k,
            template=request.template
        )
        
        result = pipeline.query(english_request.query, english_request.top_k, english_request.template)
        english_answer = result['answer']
        
        spanish_answer = translation_service.translate_response_to_spanish(english_answer)
        translation_time = time.time() - translation_start
        
        return QuerySpanishResponse(
            query=spanish_answer,
            original_query=original_query,
            english_query=english_query,
            answer=spanish_answer,
            english_answer=english_answer,
            total_documents=result['total_documents'],
            retrieval_time=result['retrieval_time'],
            translation_time=translation_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    return pipeline.get_database_stats()


@app.post("/rebuild-index")
async def rebuild_index():
    try:
        result = pipeline.build_index()
        return {"message": "Index rebuilt successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentResponse])
async def get_documents():
    try:
        loader = FileSystemLoader(settings)
        documents = loader.load_documents()
        
        return [
            DocumentResponse(
                id=doc['id'],
                filename=doc['metadata']['filename'],
                content=doc['content'],
                metadata=doc['metadata']
            )
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    try:
        loader = FileSystemLoader(settings)
        documents = loader.load_documents()
        
        document = next((doc for doc in documents if doc['id'] == document_id), None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        return DocumentResponse(
            id=document['id'],
            filename=document['metadata']['filename'],
            content=document['content'],
            metadata=document['metadata']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/{document_id}/chunks", response_model=list[ChunkResponse])
async def chunk_document(document_id: str, request: ChunkingRequest):
    try:
        loader = FileSystemLoader(settings)
        documents = loader.load_documents()
        
        document = next((doc for doc in documents if doc['id'] == document_id), None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        cleaner = DocumentCleaner()
        cleaned_doc = cleaner.clean(document)
        
        chunker = ChunkerFactory.create_chunker(request.strategy, settings)
        chunks = chunker.chunk(cleaned_doc)
        
        return [
            ChunkResponse(
                id=chunk['id'],
                content=chunk['content'],
                metadata=chunk['metadata'],
                parent_document_id=chunk['metadata']['parent_document_id']
            )
            for chunk in chunks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks", response_model=list[ChunkResponse])
async def get_all_chunks():
    try:
        with pipeline.database_manager as db:
            documents = db.get_all_documents()
            
        return [
            ChunkResponse(
                id=doc['id'],
                content=doc['content'],
                metadata=doc['metadata'],
                parent_document_id=doc['metadata'].get('parent_document_id', 'unknown')
            )
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings", response_model=list[EmbeddingResponse])
async def generate_embeddings(request: EmbeddingRequest):
    try:
        encoder = DocumentEncoder(settings)
        embeddings = encoder.model.encode(request.texts)
        
        return [
            EmbeddingResponse(
                id=f"embedding_{i}",
                content=text,
                embedding=embedding.tolist(),
                metadata={"index": i}
            )
            for i, (text, embedding) in enumerate(zip(request.texts, embeddings))
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=RetrievalResponse)
async def search_documents(request: QueryRequest):
    try:
        retrieval_result = pipeline.retriever.retrieve(request.query, request.top_k or 5)
        
        return RetrievalResponse(
            query=request.query,
            documents=retrieval_result['documents'],
            execution_time=retrieval_result['execution_time'],
            total_results=retrieval_result['total_results']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexing-steps", response_model=list[IndexingStepResponse])
async def get_indexing_steps():
    try:
        import time
        steps = []
        
        start_time = time.time()
        loader = FileSystemLoader(settings)
        documents = loader.load_documents()
        step_time = time.time() - start_time
        
        steps.append(IndexingStepResponse(
            step="loading",
            status="completed",
            data={"document_count": len(documents), "documents": [doc['id'] for doc in documents[:5]]},
            execution_time=step_time
        ))
        
        start_time = time.time()
        cleaner = DocumentCleaner()
        cleaned_docs = [cleaner.clean(doc) for doc in documents]
        step_time = time.time() - start_time
        
        steps.append(IndexingStepResponse(
            step="cleaning",
            status="completed",
            data={"cleaned_count": len(cleaned_docs)},
            execution_time=step_time
        ))
        
        start_time = time.time()
        chunker = ChunkerFactory.create_chunker('markdown_header', settings)
        all_chunks = []
        for doc in cleaned_docs:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
        step_time = time.time() - start_time
        
        steps.append(IndexingStepResponse(
            step="chunking",
            status="completed",
            data={"chunk_count": len(all_chunks), "sample_chunks": [chunk['id'] for chunk in all_chunks[:3]]},
            execution_time=step_time
        ))
        
        start_time = time.time()
        encoder = DocumentEncoder(settings)
        encoded_chunks = encoder.encode_documents(all_chunks[:5])
        step_time = time.time() - start_time
        
        steps.append(IndexingStepResponse(
            step="embedding",
            status="completed",
            data={"embedded_count": len(encoded_chunks), "embedding_dimension": len(encoded_chunks[0]['embedding']) if encoded_chunks else 0},
            execution_time=step_time
        ))
        
        return steps
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/embedding-similarity")
async def get_embedding_similarity(query: str, document_id: str):
    try:
        import numpy as np
        
        encoder = DocumentEncoder(settings)
        query_embedding = encoder.encode_query(query)
        
        with pipeline.database_manager as db:
            documents = db.get_all_documents()
            
        document = next((doc for doc in documents if doc['id'] == document_id), None)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_content = document['content'][:500]
        doc_embedding = encoder.encode_query(doc_content)
        
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        return {
            "query": query,
            "document_id": document_id,
            "similarity": float(similarity),
            "document_preview": doc_content[:100] + "...",
            "query_embedding_sample": query_embedding.tolist()[:5],
            "document_embedding_sample": doc_embedding.tolist()[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluation/generate-dataset")
async def generate_evaluation_dataset(samples: int = 30):
    try:
        generator = SyntheticDatasetGenerator(settings)
        
        with pipeline.database_manager as db:
            qa_pairs, generation_stats = generator.generate_dataset_with_stats(db, samples)
        
        filtered_pairs = generator.filter_dataset(qa_pairs)
        
        if not filtered_pairs:
            filtered_pairs = generator.filter_dataset(qa_pairs, min_groundedness=3, min_relevance=3, min_standalone=3)
        
        dataset_path = "data/evaluation/synthetic_dataset.json"
        generator.save_dataset(filtered_pairs, dataset_path)
        
        return {
            "message": "Synthetic dataset generated successfully",
            "total_generated": len(qa_pairs),
            "filtered_count": len(filtered_pairs),
            "dataset_path": dataset_path,
            "generation_stats": generation_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluation/benchmark")
async def run_benchmark_evaluation(quick: bool = True):
    try:
        dataset_path = "data/evaluation/synthetic_dataset.json"
        
        if not Path(dataset_path).exists():
            raise HTTPException(status_code=404, detail="Synthetic dataset not found. Generate it first.")
        
        generator = SyntheticDatasetGenerator(settings)
        qa_dataset = generator.load_dataset(dataset_path)
        
        benchmark = RAGBenchmark(settings)
        
        if quick:
            chunking_strategies = ["markdown_header"]
            chunk_sizes = [500, 1000]
            retrieval_strategies = ["semantic"]
            top_k_values = [3, 5]
        else:
            chunking_strategies = ["markdown_header", "sentence"]
            chunk_sizes = [500, 1000, 2000]
            retrieval_strategies = ["semantic", "hybrid"]
            top_k_values = [3, 5, 10]
        
        results = benchmark.run_grid_search(
            qa_dataset=qa_dataset,
            chunking_strategies=chunking_strategies,
            chunk_sizes=chunk_sizes,
            retrieval_strategies=retrieval_strategies,
            top_k_values=top_k_values
        )
        
        output_dir = "data/evaluation/benchmark_results"
        benchmark.save_results(results, output_dir)
        
        return {
            "message": "Benchmark evaluation completed",
            "configurations_tested": len(results),
            "results_dir": output_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/report")
async def get_evaluation_report():
    try:
        results_dir = "data/evaluation/benchmark_results"
        
        if not Path(results_dir).exists():
            raise HTTPException(status_code=404, detail="Benchmark results not found. Run benchmark first.")
        
        benchmark = RAGBenchmark(settings)
        results = benchmark.load_results(results_dir)
        
        if not results:
            raise HTTPException(status_code=404, detail="No benchmark results found")
        
        reporter = BenchmarkReporter()
        reporter.add_results(results)
        
        summary_report = reporter.generate_summary_report()
        
        return summary_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluation/detailed-analysis")
async def get_detailed_analysis():
    try:
        results_dir = "data/evaluation/benchmark_results"
        
        if not Path(results_dir).exists():
            raise HTTPException(status_code=404, detail="Benchmark results not found")
        
        benchmark = RAGBenchmark(settings)
        results = benchmark.load_results(results_dir)
        
        if not results:
            raise HTTPException(status_code=404, detail="No benchmark results found")
        
        reporter = BenchmarkReporter()
        reporter.add_results(results)
        
        detailed_analysis = reporter.generate_detailed_analysis()
        
        return detailed_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
