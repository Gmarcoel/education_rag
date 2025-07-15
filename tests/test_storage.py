import tempfile
import shutil
from pathlib import Path

from config.settings import Settings
from src.storage.database import DatabaseManager


class TestStorage:
    
    def test_database_manager(self):
        settings = Settings()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.DATABASE_PERSIST_DIR = temp_dir
            
            with DatabaseManager(settings) as db:
                assert db.get_document_count() == 0
                
                test_documents = [
                    {
                        'id': 'test1',
                        'content': 'Test document 1',
                        'embedding': [0.1, 0.2, 0.3] * 128,
                        'metadata': {'source': 'test'}
                    }
                ]
                
                db.add_documents(test_documents)
                assert db.get_document_count() == 1
                
                results = db.search([0.1, 0.2, 0.3] * 128, top_k=1)
                assert len(results) == 1
                assert results[0]['id'] == 'test1'
