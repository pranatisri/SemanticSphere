from fastapi import FastAPI

from api.routes import router as api_router

app = FastAPI(
    title="20 Newsgroups Semantic Search API",
    description=(
        "Semantic search over the 20 Newsgroups dataset using sentence-transformer "
        "embeddings, FAISS vector index, fuzzy c-means clustering, and a semantic cache."
    ),
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """
    Placeholder for any startup initialization.

    The heavy lifting (loading dataset, building embeddings, FAISS index,
    clustering, and cache) is done lazily on first query so that the API
    can start quickly even if artifacts are not precomputed.
    """
    # NOTE: We intentionally keep this lightweight. The first /query call
    # will trigger pipeline initialization if needed.
    pass


app.include_router(api_router)

