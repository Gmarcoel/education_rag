from abc import ABC, abstractmethod
import re

from config.settings import Settings
from config.constants import MIN_CHUNK_SIZE, MAX_CHUNK_SIZE


class ChunkingStrategy(ABC):
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    @abstractmethod
    def chunk(self, document: dict[str, object]) -> list[dict[str, object]]:
        pass
    
    def _create_chunk(self, content: str, document: dict[str, object], chunk_index: int) -> dict[str, object]:
        return {
            'id': f"{document['id']}_chunk_{chunk_index}",
            'content': content,
            'metadata': {
                **document['metadata'],
                'chunk_index': chunk_index,
                'chunk_size': len(content),
                'parent_document_id': document['id']
            }
        }


class FixedSizeChunker(ChunkingStrategy):
    
    def chunk(self, document: dict[str, object]) -> list[dict[str, object]]:
        content = document['content']
        chunk_size = self.settings.CHUNK_SIZE
        overlap = self.settings.CHUNK_OVERLAP
        
        if len(content) <= chunk_size:
            return [self._create_chunk(content, document, 0)]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]
            
            if len(chunk_content.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(self._create_chunk(chunk_content, document, chunk_index))
                chunk_index += 1
            
            if end >= len(content):
                break
            
            start = end - overlap
        
        return chunks


class SentenceChunker(ChunkingStrategy):
    
    def chunk(self, document: dict[str, object]) -> list[dict[str, object]]:
        content = document['content']
        sentences = self._split_into_sentences(content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.settings.CHUNK_SIZE and current_chunk:
                chunk_content = ' '.join(current_chunk)
                if len(chunk_content.strip()) >= MIN_CHUNK_SIZE:
                    chunks.append(self._create_chunk(chunk_content, document, chunk_index))
                    chunk_index += 1
                
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(self._create_chunk(chunk_content, document, chunk_index))
        
        return chunks
    
    def _split_into_sentences(self, content: str) -> list[str]:
        sentences = re.split(r'[.!?]+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        overlap_chars = self.settings.CHUNK_OVERLAP
        overlap_sentences = []
        char_count = 0
        
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence)
            else:
                break
        
        return overlap_sentences


class MarkdownHeaderChunker(ChunkingStrategy):
    
    def chunk(self, document: dict[str, object]) -> list[dict[str, object]]:
        content = document['content']
        
        if document['metadata']['extension'] != '.md':
            return FixedSizeChunker(self.settings).chunk(document)
        
        sections = self._split_by_headers(content)
        chunks = []
        chunk_index = 0
        
        for section in sections:
            if len(section['content'].strip()) >= MIN_CHUNK_SIZE:
                if len(section['content']) > self.settings.CHUNK_SIZE:
                    sub_chunks = self._split_large_section(section, document, chunk_index)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                else:
                    chunk = self._create_chunk(section['content'], document, chunk_index)
                    chunk['metadata']['header'] = section['header']
                    chunk['metadata']['header_level'] = section['level']
                    chunks.append(chunk)
                    chunk_index += 1
        
        return chunks
    
    def _split_by_headers(self, content: str) -> list[dict[str, object]]:
        lines = content.split('\n')
        sections = []
        current_section = {'header': '', 'level': 0, 'content': ''}
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if header_match:
                if current_section['content'].strip():
                    sections.append(current_section)
                
                level = len(header_match.group(1))
                header = header_match.group(2)
                current_section = {'header': header, 'level': level, 'content': ''}
            else:
                current_section['content'] += line + '\n'
        
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _split_large_section(self, section: dict[str, object], document: dict[str, object], start_index: int) -> list[dict[str, object]]:
        temp_doc = {
            'id': f"{document['id']}_section",
            'content': section['content'],
            'metadata': document['metadata']
        }
        
        sub_chunks = FixedSizeChunker(self.settings).chunk(temp_doc)
        
        for i, chunk in enumerate(sub_chunks):
            chunk['id'] = f"{document['id']}_chunk_{start_index + i}"
            chunk['metadata']['header'] = section['header']
            chunk['metadata']['header_level'] = section['level']
            chunk['metadata']['parent_document_id'] = document['id']
        
        return sub_chunks


class ChunkerFactory:
    
    @staticmethod
    def create_chunker(strategy: str, settings: Settings) -> ChunkingStrategy:
        strategies = {
            'fixed_size': FixedSizeChunker,
            'sentence': SentenceChunker,
            'markdown_header': MarkdownHeaderChunker
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        return strategies[strategy](settings)
