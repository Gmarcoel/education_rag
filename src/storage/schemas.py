from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, object]
    embedding: list[float] | None = None
    created_at: datetime | None = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class SearchResult:
    document: Document
    score: float
    distance: float | None = None


@dataclass
class QueryResult:
    query: str
    results: list[SearchResult]
    total_results: int
    execution_time: float
