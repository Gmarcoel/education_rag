import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import requests
import json
import time

def test_endpoints():
    base_url = "http://localhost:8000"
    
    print("Testing API endpoints...")
    
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        
        response = requests.get(f"{base_url}/documents")
        if response.status_code == 200:
            docs = response.json()
            print(f"✅ Documents endpoint: {len(docs)} documents found")
            
            if docs:
                doc_id = docs[0]['id']
                
                response = requests.get(f"{base_url}/documents/{doc_id}")
                print(f"✅ Single document endpoint: {response.status_code}")
                
                chunk_request = {"document_id": doc_id, "strategy": "markdown_header"}
                response = requests.post(f"{base_url}/documents/{doc_id}/chunks", 
                                       json=chunk_request)
                if response.status_code == 200:
                    chunks = response.json()
                    print(f"✅ Chunking endpoint: {len(chunks)} chunks created")
                
                similarity_url = f"{base_url}/embedding-similarity?query=programming&document_id={doc_id}"
                response = requests.get(similarity_url)
                print(f"✅ Similarity endpoint: {response.status_code}")
        
        response = requests.get(f"{base_url}/chunks")
        if response.status_code == 200:
            chunks = response.json()
            print(f"✅ All chunks endpoint: {len(chunks)} chunks found")
        
        response = requests.get(f"{base_url}/indexing-steps")
        if response.status_code == 200:
            steps = response.json()
            print(f"✅ Indexing steps endpoint: {len(steps)} steps found")
        
        embed_request = {"texts": ["What is programming?", "How do variables work?"]}
        response = requests.post(f"{base_url}/embeddings", json=embed_request)
        if response.status_code == 200:
            embeddings = response.json()
            print(f"✅ Embeddings endpoint: {len(embeddings)} embeddings generated")
        
        search_request = {"query": "object-oriented programming", "top_k": 3}
        response = requests.post(f"{base_url}/search", json=search_request)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Search endpoint: {results['total_results']} results found")
        
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats endpoint: {stats['document_count']} documents in DB")
        
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: uv run python api/server.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_endpoints()
