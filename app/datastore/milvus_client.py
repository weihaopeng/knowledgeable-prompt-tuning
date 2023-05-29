import os
from app.datastore.providers.milvus_datastore import MilvusDataStore
from dotenv import load_dotenv

load_dotenv('./.env.example')

db_host = os.getenv("APP_DB_HOST", "localhost")
db_port = int(os.getenv("APP_DB_PORT", 19530))
store_name = os.getenv("APP_DB_STORE_NAME", "knowledge_example")

data_store = MilvusDataStore(milvus_host = db_host, milvus_port = db_port)

if data_store._has_collection(store_name) is False:
    data_store.create(store_name)
