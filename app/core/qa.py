from pydantic import BaseModel
from typing import List, TypedDict, Optional
import argparse
import re
import time

from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from app.datastore.vectorstore import MilvusStore
from app.datastore.milvus_client import data_store
from app.datastore.vectorstore import MilvusStore
from app.core.vector_db_chain import MilvusVectorDBQAWithSourcesChain

import os

store_name = os.getenv("APP_DB_STORE_NAME", "knowledge_example")

class ContentType(TypedDict):
   id: str
   text: str

class QaWorker (BaseModel):

    def __init__(self):
        super().__init__()


    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def qa(self, question: str, text_ids: Optional[List[str]] = None):
        parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
        parser.add_argument('--question', type=str, default=question, help='The question to ask the notion DB', )

        args = parser.parse_args()

        embeddings = OpenAIEmbeddings() # type: ignore
        store = MilvusStore(embeddings, data_store)

        llm = OpenAI(temperature=0) # type: ignore
        chain = MilvusVectorDBQAWithSourcesChain.from_llm(llm, vectorstore=store)
        inputs = {"question": args.question, "text_ids": text_ids, "store_name": store_name, "token_max":4096}
        result = chain(inputs)

        docs = chain._get_docs(inputs)

        result['source'] = list(map(lambda doc: doc.metadata['source'], docs))
        result['source_id'] = list(map(lambda doc: doc.metadata['source_id'], docs))   


        return {key: value for key, value in result.items() if key in ['answer', 'source', 'source_id']}




qaWoker = QaWorker()

