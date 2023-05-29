from pydantic import BaseModel, Field
from typing import List, TypedDict, Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from app.datastore.vectorstore import MilvusStore
from app.datastore.milvus_client import data_store

import os

store_name = os.getenv("APP_DB_STORE_NAME", "knowledge_example")

class ContentType(TypedDict):
   id: Optional[str]
   text: str


class EmbeddingWorker (BaseModel):
    chunk_size: int = Field(1250)
    chunk_overlap: int = Field(150)

    def __init__(self, chunk_size: int = 1250, chunk_overlap: int = 150):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    async def embedding(self, contents: List[ContentType]):
        
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator="\n\n")
        docs = []
        for i, content in enumerate(contents):
            text_id = str(content['id'])
            splits = text_splitter.split_text(content['text'])
            for j, split in enumerate(splits):
                unit = { 'text': split, 'text_id': text_id }
                print(unit)
                docs.append(unit)

        embeddings = OpenAIEmbeddings() # type: ignore
        store = MilvusStore(embeddings, data_store)
        store.upsert(store_name = store_name, documents = docs)
        return 'ok'


    def deleteEmbedding(self, ids: List[str]):
        data_store.delete(store_name=store_name, ids=ids)
        return 'ok'



embeddingWoker = EmbeddingWorker(chunk_size = 1250, chunk_overlap = 150)

