import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Settings
from scripts.demo import RAGPipeline


def test_retrieval():
    settings = Settings()
    pipeline = RAGPipeline(settings)
    
    query = "What is object-oriented programming?"
    
    print(f"Testing retrieval for: {query}")
    
    retrieval_result = pipeline.retriever.retrieve(query, 3)
    documents = retrieval_result['documents']
    
    print(f"Retrieved {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"ID: {doc['id']}")
        print(f"Content preview: {doc['content'][:200]}...")
        print(f"Source: {doc['metadata'].get('source', 'Unknown')}")
        if 'distance' in doc:
            print(f"Distance: {doc['distance']:.4f}")
    
    return documents


if __name__ == "__main__":
    test_retrieval()
