import pytest
from pathlib import Path
import tempfile
import shutil

from config.settings import Settings
from src.ingestion.loader import FileSystemLoader, MarkdownLoader, TextLoader


class TestDocumentLoaders:
    
    def test_markdown_loader(self):
        loader = MarkdownLoader()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nThis is a test.")
            temp_path = Path(f.name)
        
        try:
            assert loader.supports(temp_path)
            content = loader.load(temp_path)
            assert "# Test Document" in content
            assert "This is a test." in content
        finally:
            temp_path.unlink()
    
    def test_text_loader(self):
        loader = TextLoader()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a text file.")
            temp_path = Path(f.name)
        
        try:
            assert loader.supports(temp_path)
            content = loader.load(temp_path)
            assert content == "This is a text file."
        finally:
            temp_path.unlink()
    
    def test_filesystem_loader(self):
        settings = Settings()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.DATA_RAW_DIR = temp_dir
            
            test_file = Path(temp_dir) / "test.md"
            test_file.write_text("# Test\n\nContent here.")
            
            loader = FileSystemLoader(settings)
            documents = loader.load_documents()
            
            assert len(documents) == 1
            assert documents[0]['content'] == "# Test\n\nContent here."
            assert documents[0]['metadata']['filename'] == "test.md"
