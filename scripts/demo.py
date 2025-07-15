import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from pathlib import Path

from config.settings import Settings
from src.ingestion.loader import FileSystemLoader
from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.chunker import ChunkerFactory
from src.embedding.encoder import DocumentEncoder
from src.embedding.cache import EmbeddingCache
from src.storage.database import DatabaseManager
from src.retrieval.retriever import RetrieverFactory
from src.retrieval.reranker import RerankerFactory
from src.generation.llm import LLMService
from src.generation.prompt_templates import PromptBuilder


class RAGPipeline:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._initialize_components()
    
    def _initialize_components(self):
        self.loader = FileSystemLoader(self.settings)
        self.cleaner = DocumentCleaner()
        self.chunker = ChunkerFactory.create_chunker('markdown_header', self.settings)
        self.encoder = DocumentEncoder(self.settings)
        self.cache = EmbeddingCache(self.settings)
        self.database_manager = DatabaseManager(self.settings)
        self.retriever = RetrieverFactory.create_retriever(
            'semantic', self.database_manager.database, self.encoder, self.settings
        )
        self.reranker = RerankerFactory.create_reranker('distance', self.settings)
        self.llm_service = LLMService(self.settings)
        self.prompt_builder = PromptBuilder()
    
    def build_index(self) -> dict[str, object]:
        print("Loading documents...")
        documents = self.loader.load_documents()
        
        print(f"Loaded {len(documents)} documents")
        print("Cleaning documents...")
        cleaned_documents = [self.cleaner.clean(doc) for doc in documents]
        
        print("Chunking documents...")
        all_chunks = []
        for doc in cleaned_documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        print("Generating embeddings...")
        
        cached_chunks = self.cache.load_batch_embeddings(all_chunks, self.settings.EMBEDDING_MODEL)
        chunks_to_encode = [chunk for chunk in cached_chunks if 'embedding' not in chunk]
        
        if chunks_to_encode:
            encoded_chunks = self.encoder.encode_documents(chunks_to_encode)
            self.cache.save_batch_embeddings(encoded_chunks, self.settings.EMBEDDING_MODEL)
            
            for i, chunk in enumerate(cached_chunks):
                if 'embedding' not in chunk:
                    chunk.update(encoded_chunks[chunks_to_encode.index(chunk)])
        
        print("Storing in database...")
        with self.database_manager as db:
            db.add_documents(cached_chunks)
        
        return {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "chunks_encoded": len(chunks_to_encode),
            "chunks_cached": len(all_chunks) - len(chunks_to_encode)
        }
    
    def query(self, query: str, top_k: int | None = None, 
              template: str = 'BASIC_QA') -> dict[str, object]:
        if top_k is None:
            top_k = self.settings.TOP_K
        
        print(f"Processing query: {query}")
        
        retrieval_result = self.retriever.retrieve(query, top_k)
        documents = retrieval_result['documents']
        
        if not documents:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "documents": [],
                "retrieval_time": retrieval_result['execution_time']
            }
        
        reranked_documents = self.reranker.rerank(query, documents)
        
        prompt = self.prompt_builder.build_rag_prompt(query, reranked_documents, template)
        
        print("Generating response...")
        answer = self.llm_service.generate_response(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "documents": reranked_documents,
            "retrieval_time": retrieval_result['execution_time'],
            "total_documents": len(documents)
        }
    
    def get_database_stats(self) -> dict[str, object]:
        with self.database_manager as db:
            return {
                "document_count": db.get_document_count(),
                "cache_size": self.cache.get_cache_size()
            }


def main():
    settings = Settings()
    
    if not Path(settings.DATA_RAW_DIR).exists():
        print(f"Creating data directory: {settings.DATA_RAW_DIR}")
        Path(settings.DATA_RAW_DIR).mkdir(parents=True, exist_ok=True)
        print("Please place your markdown files in the data/raw/ directory")
        return
    
    pipeline = RAGPipeline(settings)
    
    stats = pipeline.get_database_stats()
    if stats["document_count"] == 0:
        print("Building index...")
        build_result = pipeline.build_index()
        print(f"Index built: {build_result}")
    else:
        print(f"Using existing index with {stats['document_count']} documents")
    
    print("\nRAG Pipeline ready!")
    print("Enter your queries (type 'exit' to quit):")
    
    while True:
        user_query = input("\nQuery: ").strip()
        if user_query.lower() == 'exit':
            break
        
        if not user_query:
            continue
        
        result = pipeline.query(user_query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"Retrieved {result['total_documents']} documents")
        print(f"Retrieval time: {result['retrieval_time']:.3f} seconds")


if __name__ == "__main__":
    main()
