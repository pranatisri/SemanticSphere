from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.pipeline import get_global_pipeline


router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")


class QueryResult(BaseModel):
    document_id: int
    score: float
    text_snippet: str
    category: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: List[QueryResult]
    dominant_cluster: Optional[int]


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Main semantic search endpoint.

    Steps:
    1) Encode the query into an embedding.
    2) Route through the semantic cache (cluster-aware).
    3) If cache miss, hit the FAISS index and persist into cache.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    pipeline = get_global_pipeline()
    (
        cache_hit,
        matched_query,
        similarity_score,
        results,
        dominant_cluster,
    ) = pipeline.search(request.query)

    result_items: List[QueryResult] = []
    for r in results:
        result_items.append(
            QueryResult(
                document_id=int(r["doc_id"]),
                score=float(r["score"]),
                text_snippet=r["text_snippet"],
                category=r.get("category"),
            )
        )

    return QueryResponse(
        query=request.query,
        cache_hit=cache_hit,
        matched_query=matched_query,
        similarity_score=similarity_score,
        result=result_items,
        dominant_cluster=dominant_cluster,
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats_endpoint() -> CacheStatsResponse:
    """Return global cache statistics."""
    pipeline = get_global_pipeline()
    stats: Dict[str, Any] = pipeline.cache.get_stats()
    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
    )


@router.delete("/cache", response_model=CacheStatsResponse)
async def clear_cache_endpoint() -> CacheStatsResponse:
    """Clear the semantic cache and reset statistics."""
    pipeline = get_global_pipeline()
    pipeline.cache.clear()
    stats: Dict[str, Any] = pipeline.cache.get_stats()
    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
    )

