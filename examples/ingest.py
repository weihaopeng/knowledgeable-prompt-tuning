"""This is the logic for ingesting Notion data into LangChain."""

import os
import pickle
from pathlib import Path
import time

from dotenv import load_dotenv

load_dotenv('./.env.example')

import os
import openai

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from app.datastore.providers.milvus_datastore import MilvusDataStore
from app.datastore.vectorstore import MilvusStore


api_key = os.getenv("OPENAI_API_KEY", "")
api_base = os.getenv("OPENAI_API_BASE", "")

if (api_key == "") or (api_base == ""):
    raise Exception('Sorry, need api key and openai api_base!')
openai.api_base = api_base
openai.api_key = api_key


start_time = time.time()

ps = list(Path("./examples/data/").glob("*.txt"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)


# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0, separator="\n\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    print(i)
    for j, split in enumerate(splits):
        unit = {'text': split, 'text_id': str(i), 'id': j}
        print(unit)
        docs.append(unit)


embeddings = OpenAIEmbeddings() # type: ignore


data_store = MilvusDataStore(milvus_host="localhost", milvus_port=19530)
store = MilvusStore(embeddings, data_store)

data_store.drop("knowledge_example")

store.create(store_name="knowledge_example")

store.upsert(store_name="knowledge_example", documents=docs)

end_time = time.time()

print("耗时：{:.2f}秒".format(end_time - start_time))
