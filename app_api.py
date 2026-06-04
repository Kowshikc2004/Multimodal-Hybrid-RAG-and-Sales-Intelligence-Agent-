import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service.initialize_pipeline()
    yield


app = FastAPI(
    title="Jokaan RAG API",
    description="API endpoint for Jokaan RAG question answering using Chroma Cloud retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str
    include_context: bool = True
    top_k: int = 3


class ContextItem(BaseModel):
    source: str
    chunk_index: Optional[int] = None
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_count: int
    context: List[ContextItem]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "collection": os.getenv("CHROMA_DATABASE") or "Jokaan_RAG_exe",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = rag_service.answer_query(request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not request.include_context:
        result["context"] = []

    return QueryResponse(**result)


@app.post("/rebuild")
def rebuild() -> Dict[str, str]:
    try:
        rag_service.build_vector_database()
        return {"status": "rebuild started", "message": "The vector database rebuild completed successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_api:app", host="0.0.0.0", port=8000)
