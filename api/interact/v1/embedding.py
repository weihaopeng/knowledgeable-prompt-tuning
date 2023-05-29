from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, TypedDict

from app.core.embedding import embeddingWoker



class ContentType(TypedDict):
   id: str
   text: str

class EmbeddingRequest(BaseModel):
  contents: List[ContentType] = Field(..., description="知识的文本内容，每项包括id与text")


class DeleteEmbeddingRequest(BaseModel):
   content_ids: List[str] = Field(..., description="要删除的知识id集")

class EmbeddingResponse(BaseModel):
  status: str = Field(..., description="ok | error")

class ExceptionResponseSchema(BaseModel):
    error: str

embedding_router = APIRouter()

@embedding_router.post(
  "",
  response_model=EmbeddingResponse,
  responses={"400": {"model": ExceptionResponseSchema}}
)

async def Embedding(request: EmbeddingRequest):
    # return JSONResponse(content=request.contents, status_code=200)
    try:
      await embeddingWoker.embedding(contents=request.contents)
      return JSONResponse(content = { "res": 'ok' }, status_code = 200)
    except:
      raise HTTPException(status_code=400, detail=f'知识入库失败!')

