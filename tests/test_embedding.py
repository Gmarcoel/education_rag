import numpy as np

from config.settings import Settings
from src.embedding.encoder import DocumentEncoder


class TestEmbedding:
    
    def test_document_encoder(self):
        settings = Settings()
        encoder = DocumentEncoder(settings)
        
        documents = [
            {
                'id': 'doc1',
                'content': 'This is a test document.',
                'metadata': {}
            },
            {
                'id': 'doc2', 
                'content': 'This is another test document.',
                'metadata': {}
            }
        ]
        
        encoded_docs = encoder.encode_documents(documents)
        
        assert len(encoded_docs) == 2
        for doc in encoded_docs:
            assert 'embedding' in doc
            assert isinstance(doc['embedding'], list)
            assert len(doc['embedding']) > 0
    
    def test_query_encoding(self):
        settings = Settings()
        encoder = DocumentEncoder(settings)
        
        query = "What is machine learning?"
        embedding = encoder.encode_query(query)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
