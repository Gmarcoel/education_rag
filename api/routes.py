from fastapi import APIRouter, HTTPException

from api.models import QueryRequest, QueryResponse, IndexBuildRequest, IndexBuildResponse, DatabaseStatsResponse
from config.settings import Settings
from scripts.demo import RAGPipeline


router = APIRouter()
settings = Settings()
pipeline = RAGPipeline(settings)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        result = pipeline.query(
            query=request.query,
            top_k=request.top_k,
            template=request.template
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-index", response_model=IndexBuildResponse)
async def build_index(request: IndexBuildRequest) -> IndexBuildResponse:
    try:
        if request.force_rebuild:
            with pipeline.database_manager as db:
                db.clear_collection()
            pipeline.cache.clear_cache()
        
        result = pipeline.build_index()
        return IndexBuildResponse(
            success=True,
            message="Index built successfully",
            details=result
        )
    except Exception as e:
        return IndexBuildResponse(
            success=False,
            message=f"Failed to build index: {str(e)}"
        )


@router.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats() -> DatabaseStatsResponse:
    try:
        stats = pipeline.get_database_stats()
        return DatabaseStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
