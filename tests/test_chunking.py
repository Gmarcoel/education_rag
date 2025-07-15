import tempfile
from pathlib import Path

from config.settings import Settings
from src.ingestion.chunker import ChunkerFactory, MarkdownHeaderChunker


class TestChunkers:
    
    def test_markdown_header_chunker(self):
        settings = Settings()
        chunker = MarkdownHeaderChunker(settings)
        
        document = {
            'id': 'test_doc',
            'content': '''# Introduction
This is the introduction.

## Section 1
This is section 1 content.

## Section 2
This is section 2 content.''',
            'metadata': {'extension': '.md'}
        }
        
        chunks = chunker.chunk(document)
        assert len(chunks) >= 2
        
        for chunk in chunks:
            assert 'header' in chunk['metadata']
            assert 'header_level' in chunk['metadata']
    
    def test_chunker_factory(self):
        settings = Settings()
        
        chunker = ChunkerFactory.create_chunker('fixed_size', settings)
        assert chunker is not None
        
        chunker = ChunkerFactory.create_chunker('sentence', settings)
        assert chunker is not None
        
        chunker = ChunkerFactory.create_chunker('markdown_header', settings)
        assert chunker is not None
