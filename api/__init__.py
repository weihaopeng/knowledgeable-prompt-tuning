from fastapi import APIRouter

from api.interact.v1.embedding import embedding_router as embedding_v1_router
from api.interact.v1.qa import qa_router as qa_v1_router

router = APIRouter()
router.include_router(embedding_v1_router, prefix="/api/interact/v1/embeddings", tags=["embedding"])
router.include_router(qa_v1_router, prefix="/api/interact/v1/qa", tags=["qa"])

__all__ = ["router"]
