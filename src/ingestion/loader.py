from abc import ABC, abstractmethod
from pathlib import Path
import os

from config.settings import Settings
from config.constants import SUPPORTED_FILE_EXTENSIONS


class DocumentLoader(ABC):
    
    @abstractmethod
    def load(self, file_path: Path) -> str:
        pass
    
    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        pass


class MarkdownLoader(DocumentLoader):
    
    def load(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.md'


class TextLoader(DocumentLoader):
    
    def load(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.txt'


class DocumentLoaderFactory:
    
    def __init__(self):
        self._loaders: list[DocumentLoader] = [
            MarkdownLoader(),
            TextLoader()
        ]
    
    def get_loader(self, file_path: Path) -> DocumentLoader:
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader
        raise ValueError(f"No loader found for file: {file_path}")
    
    def register_loader(self, loader: DocumentLoader):
        self._loaders.append(loader)


class FileSystemLoader:
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loader_factory = DocumentLoaderFactory()
    
    def load_documents(self) -> list[dict[str, object]]:
        documents = []
        data_dir = Path(self.settings.DATA_RAW_DIR)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        for file_path in self._get_supported_files(data_dir):
            try:
                loader = self.loader_factory.get_loader(file_path)
                content = loader.load(file_path)
                
                document = {
                    'id': self._generate_document_id(file_path),
                    'content': content,
                    'metadata': {
                        'source': str(file_path),
                        'filename': file_path.name,
                        'extension': file_path.suffix,
                        'size': os.path.getsize(file_path)
                    }
                }
                documents.append(document)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return documents
    
    def _get_supported_files(self, directory: Path) -> list[Path]:
        files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS:
                files.append(file_path)
        return files
    
    def _generate_document_id(self, file_path: Path) -> str:
        return f"doc_{file_path.stem}_{hash(str(file_path)) % 10000}"
