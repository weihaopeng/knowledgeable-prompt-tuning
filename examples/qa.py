"""Ask a question to the notion database."""
import os
# import faiss
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import VectorDBQAWithSourcesChain

import argparse
import time

from app.datastore.providers.milvus_datastore import MilvusDataStore
from app.datastore.vectorstore import MilvusStore
from app.core.vector_db_chain import MilvusVectorDBQAWithSourcesChain
import openai
from dotenv import load_dotenv

load_dotenv('./.env.example')

api_key = os.getenv("OPENAI_API_KEY", "")
api_base = os.getenv("OPENAI_API_BASE", "")

if (api_key == "") or (api_base == ""):
    raise Exception('Sorry, need api key and openai api_base!')
openai.api_base = api_base
openai.api_key = api_key


start_time = time.time()

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('--question', type=str, default="汉堡王在2019年发生了什么事", help='The question to ask the notion DB', )
args = parser.parse_args()

data_store = MilvusDataStore(milvus_host="localhost", milvus_port=19530)
embeddings = OpenAIEmbeddings() # type: ignore
store = MilvusStore(embeddings, data_store)
chain = MilvusVectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store) # type: ignore
result = chain({"question": args.question, "text_ids": None, "store_name": "knowledge_example"})
print(f"\n\nAnswer: {result['answer']}")
print(f"Sources: {result['sources']}")

end_time = time.time()

print("耗时：{:.2f}秒".format(end_time - start_time))
