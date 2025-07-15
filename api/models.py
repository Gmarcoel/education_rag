from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    template: str = 'BASIC_QA'


class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: list[dict[str, object]]
    retrieval_time: float
    total_documents: int


class IndexBuildRequest(BaseModel):
    force_rebuild: bool = False


class IndexBuildResponse(BaseModel):
    success: bool
    message: str
    details: dict[str, object] | None = None


class DatabaseStatsResponse(BaseModel):
    document_count: int
    cache_size: int

