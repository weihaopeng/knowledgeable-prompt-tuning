from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.qa import qaWoker

from typing import List, TypedDict, Optional

class ContentType(TypedDict):
   id: str
   text: str


class QARequest(BaseModel):
  text_ids: Optional[List[str]] = None # "讲品对应的话术集合"
  question: str = Field(..., description="当前QA的问题")
  need_source: Optional[bool] = None # debug时使用
  # streaming: Optional[bool] = None # map-reduce，暂时无法streaming


class QAResponse(BaseModel):
  answer: str = Field(..., description="回复文案")
  source: Optional[str] = Field(..., description="回复对应的讲品话术原文")
  source_id: Optional[str] = Field(..., description="回复对应的讲品话术id")


class ExceptionResponseSchema(BaseModel):
    error: str

qa_router = APIRouter()

@qa_router.post("",
    response_model=QAResponse,
    responses={"400": {"model": ExceptionResponseSchema}})
def Qa(request: QARequest):
    try:
        res = qaWoker.qa(text_ids=request.text_ids, question=request.question)
        if request.need_source:
            pass
        else:
            res = { "answer": res["answer"] } # type: ignore
        return JSONResponse(content = res, status_code = 200)
    except Exception as err:
        print(err)
        raise HTTPException(status_code=400, detail=f'获取回复失败!')
