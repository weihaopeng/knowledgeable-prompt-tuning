import json
import datetime
import time
from typing import Text, Tuple, List, Dict, Callable, Optional, Union, Any

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

from app.datastore.datastore import DataStore
from app.middleware.log import logger

EMBEDDING_FIELD = "embedding"
UPSERT_BATCH_SIZE = 100
OUTPUT_DIM = 1536
INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

class Required:
    pass

SCHEMA = [
    (
        "id",
        FieldSchema(
            name="id",
            description="主键",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),
        Required,
    ),
    (
        EMBEDDING_FIELD,
        FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=OUTPUT_DIM),
        Required,
    ),
    (
        "text_id",
        FieldSchema(name="text_id", description="讲品id", dtype=DataType.VARCHAR, max_length=65535),
        Required,
    ),
    (
        "text",
        FieldSchema(name="text", description="讲品原文", dtype=DataType.VARCHAR, max_length=65535),
        "",
    ),
    ("created_at", FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=65535), Required),

]

class MilvusDataStore(DataStore):
    def __init__(self, milvus_host: Text, milvus_port: int):
        self.host = milvus_host
        self.port = milvus_port
        # TODO: delete index from buffer if buffer exceeds memory limit
        # 目前这里的buffer并没有实现真正的数据库缓冲区机制，只是一个普通的字典，用来储存Collection实例而已
        self.buffer = {}
        self._create_connection()

    @classmethod
    def _get_schema(cls, contain_auto_id = False):
        if False is contain_auto_id:
            sli = slice(1, len(SCHEMA))
            return SCHEMA[sli]
        return SCHEMA

    def _create_connection(self) -> None:
        """
        create connection to milvus server
        """
        try:
            alias = ""
            # Check if the connection already exists
            for x in connections.list_connections():
                addr = connections.get_connection_addr(x[0])
                if x[1] and ('address' in addr) and (addr['address'] == "{}:{}".format(self.host, self.port)):
                    alias = x[0]
                    logger.info("Reuse connection to Milvus server '{}:{}' with alias '{:s}'"
                                     .format(self.host, self.port, alias))
                    break

            # Connect to the Milvus instance using the host and port
            if len(alias) == 0:
                connections.connect(
                    host=self.host,
                    port=self.port
                )
                logger.info("Create connection to Milvus server '{}:{}' with alias '{:s}'"
                                 .format(self.host, self.port, alias))
        except Exception as e:
            logger.error("Failed to create connection to Milvus server '{}:{}', error: {}"
                            .format(self.host, self.port, e))

    @staticmethod
    def _has_collection(name: Text):
        return utility.has_collection(name)

    def create(self, store_name: Text) -> bool:
        self._create_connection()
        return self._create_collection(store_name)

    def _create_collection(self, collection_name: Text) -> bool:
        """Create a collection.

        Args:
            collection_name (str): collection name.
        """
        status = True
        try:
            # If the collection exists, don't create collection
            if self._has_collection(collection_name):
                logger.error(f"collection {collection_name} already exists, create collection fail.")
                status = False
                return status

            schema = [field[1] for field in SCHEMA]
            schema = CollectionSchema(fields=schema)
            # Use the schema to create a new collection
            collection = Collection(
                name=collection_name,
                schema=schema
            )
            index_status = self._create_index(collection)
            if index_status:
                if collection_name in self.buffer:
                    status = False
                    logger.error(f"collection name already in buffer, please check your code!!!")
                else:
                    logger.info("Create Milvus collection '{}'".format(collection_name))
            else:
                status = False

        except Exception as e:
            logger.error("Failed to create collection '{}', error: {}".format(collection_name, e))
            status = False
        return status

    def _create_index(self, collection: Collection) -> bool:
        """
        create index for some fields in collection
        :param collection: the collection in which we create index
        :return:
        """
        # TODO: verify index/search params passed by os.environ
        status = True
        try:
            i_p = INDEX_PARAMS
            logger.info("Attempting creation of Milvus '{}' index".format(i_p["index_type"]))
            collection.create_index(EMBEDDING_FIELD, i_p)
            collection.create_index(
                field_name="text_id",
                index_name="text_id_index",
            )
            logger.info("Creation of Milvus '{}' index successful".format(i_p["index_type"]))

        except Exception as e:
            logger.error("Failed to create index, error: {}".format(e))
            status = False

        return status

    def upsert(
        self, store_name: Text, documents: List[Dict]
    ) -> bool:
        self._create_connection()
        status = True
        text_ids = [item["text_id"] for item in documents]
        delete_status = self.delete(
            store_name,
            ids=text_ids,
            delete_all=False
        )
        # Delete any existing vectors for documents with the input document ids
        if delete_status:
            upsert_status = self._upsert(store_name, documents)
            if not upsert_status:
                status = False
        else:
             status = False
        return status

    def _upsert(self, store_name: Text, documents: List[Dict]) -> bool:
        """Upsert documents into the datastore.

        Args:
            store_name (str): collection name
            documents (List[Dict]): A list of documents to insert

        Raises:
            e: Error in upserting data.

        Returns:
            bool: _upsert status.
            [
                {'id': '0_0', 'vector': [0.2233, 0.1234, 0.0156 ..., 0.0890], 'text': '天王盖地虎'},
                {'id': '1_1', 'vector': [0.6729, 0.0211, -0.0529 ..., 0.0984], 'text': '轻轻地放下这个杯子吧，老爹。'},
                {'id': '1_2', 'vector': [-0.0023, 0.0372, 0.0333 ..., 0.0741], 'text': 'a squat man is staggering out the arcade.'},
            ]
        """
        status = True
        try:
            # List to collect all the insert data
            insert_data = [[] for _ in range(len(self._get_schema()))]

            # Go through each document chunklist and grab the data
            for document in documents:
                # Extract data from the chunk
                list_of_data = self._get_values(document)
                # Append each field to the insert_data
                for i in range(len(insert_data)):
                    insert_data[i].append(list_of_data[i])
            # Slice up our insert data into batches
            batches = [
                [insert_data[j][i: i + UPSERT_BATCH_SIZE]
                for j in range(len(insert_data))]
                for i in range(0, len(insert_data[0]), UPSERT_BATCH_SIZE)
            ]

            # get collection
            collection = Collection(store_name) # type: ignore

            # Attempt to insert each batch into collection
            for batch in batches:
                if len(batch[0]) != 0:
                    collection.insert(batch)
            logger.info(f"Upserted successfully, {len(insert_data[0])} data in total.")

            # wait for the operation to work
            time.sleep(1.5)

            # This setting performs flushes after insert. Small insert == bad to use
            # self.col.flush()

        except Exception as e:
            logger.error("Failed to insert records, error: {}".format(e))
            status = False
            raise e
        return status

    def _query(
        self,
        store_name: Text,
        queries: List[Dict],
    ) -> Optional[List[Dict]]:
        """Query the with hybrid search.

        Search the embedding and its filter in the collection.

        Args:
            store_name (Text):
            queries (List[QueryWithEmbedding]): The list of searches to perform.

        Returns:
            Optional[List[Dict]]: Results for each search.
        """
        # Async to perform the query, adapted from pinecone implementation
        def _single_query(collection_name: Text, query: Dict) -> Dict:
            try:
                # filter = None
                # # Set the filter to expression that is valid for Milvus
                # if query["text_ids"] is not None:
                #     # Either a valid filter or None will be returned
                #     filter = self._get_filter(query["text_ids"])
                filter = self._get_filter(query)

                # Perform our search
                output_fields = self._get_output_fields()
                if collection_name in self.buffer:
                    collection = self.buffer.get(collection_name)
                else:
                    collection = Collection(collection_name) # type: ignore
                    self.load_to_buffer(collection_name, collection)
                res = collection.search( # type: ignore
                    data=[query["embedding"]],
                    anns_field=EMBEDDING_FIELD,
                    param=SEARCH_PARAMS,
                    limit=query["top_k"],
                    expr=filter,
                    output_fields=output_fields
                )
                # Results that will hold our DocumentChunkWithScores
                results = []
                # Parse every result for our search
                for hit in res[0]:  # type: ignore
                    # The distance score for the search result, falls under DocumentChunkWithScore
                    score = hit.score
                    # Our metadata info, falls under DocumentChunkMetadata
                    metadata = {}
                    # Grab the values that correspond to our fields, ignore pk and embedding.
                    for x in output_fields:
                        metadata[x] = hit.entity.get(x)

                    # collection wanted data
                    res_unit = {
                                # "id": metadata["id"],
                                "text_id": metadata["text_id"],
                                "text": metadata["text"],
                                "score": score
                               }

                    results.append(res_unit)

                final_res = {"query": query["query"], "results": results}
                return final_res

            except Exception as e:
                logger.error("Failed to query, error: {}".format(e))
                return {"query": query["query"], "results": []}
        self._create_connection()
        collection_name = store_name
        if not self._is_collection_exists(collection_name):
            logger.error(f"collection {collection_name} not exists, query failed.")
            return None
        # TODO: concurrent this with multi-threads or asyncio
        outputs: List[Dict] = [_single_query(collection_name, query) for query in queries]

        return outputs

    def delete(
        self,
        store_name: Text,
        ids: Optional[List[str]] = None,
        text_ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """Delete the entities based either on the chunk_id of the vector,

        Args:
            store_name (Text):
            ids (Optional[List[str]], optional): The document_ids to delete. Defaults to None.
            filter (Optional[DocumentMetadataFilter], optional): The filter to delete by. Defaults to None.
            delete_all (Optional[bool], optional): Whether to drop the collection and recreate it. Defaults to None.
        """
        status = True
        try:
            self._create_connection()
            collection_name = store_name
            if not self._is_collection_exists(collection_name):
                logger.error(f"collection {collection_name} not exists, deletion failed.")
                status = False
                return status
            # If deleting all, drop and create the new collection
            if delete_all:
                # if collection in buffer
                in_buffer = True
                if collection_name in self.buffer:
                    # release collection
                    collection = self.release_from_buffer(collection_name)
                else:
                    in_buffer = False
                    collection = Collection(collection_name) # type: ignore
                # Drop the collection
                utility.drop_collection(collection_name)
                create_status = self._create_collection(collection_name)
                if not create_status:
                    status = False
                    logger.error(f"Fail to delete all data from collection {collection_name}")
                else:
                    logger.info(f"Delete the entire collection {collection_name}")
                    if in_buffer:
                        collection = Collection(collection_name) # type: ignore
                        self.load_to_buffer(collection_name, collection)
                return status

            # Keep track of how many we have deleted for later printing
            delete_count = 0
            batch_size = 100
            pk_name = "id"

            # According to the api design, the ids is a list of text_ids,
            # text_id is not primary key, use query+delete to workaround,
            if (ids is not None) and len(ids) > 0:
                # Add quotation marks around the string format id
                ids = ['"' + str(id) + '"' for id in ids]

                in_buffer = True
                # get collection
                collection = self.buffer.get(collection_name, None)
                # if collection not in buffer, load the collection to buffer
                if collection is None:
                    in_buffer = False
                    collection = Collection(collection_name) # type: ignore
                    self.load_to_buffer(collection_name, collection)

                # Query for the pk's of entries that match id's
                text_ids = collection.query(f"text_id in [{','.join(ids)}]")
                # Convert to list of pks
                pks = [str(entry[pk_name]) for entry in text_ids]  # type: ignore
                # # rewrite the expression
                # pks = ['"' + pk + '"' for pk in pks]
                # don't need rewrite, id is Int64

                # Delete by ids batch by batch(avoid too long expression)
                logger.info("Apply {:d} deletions".format(len(pks)))
                while len(pks) > 0:
                    batch_pks = pks[:batch_size]
                    pks = pks[batch_size:]
                    # Delete the entries batch by batch
                    res = collection.delete(f"{pk_name} in [{','.join(batch_pks)}]")
                    # Increment our deleted count
                    delete_count += int(res.delete_count)  # type: ignore

                # if collection not in buffer, release collection
                if not in_buffer:
                    _ = self.release_from_buffer(collection_name)

                logger.info(f"{delete_count} records deleted from collection {store_name}.")
            # wait for the operation to work
            time.sleep(1.5)

        except Exception as e:
            logger.error("Failed to delete by ids, error: {}".format(e))
            status = False

        # This setting performs flushes after delete. Small delete == bad to use
        # self.col.flush()

        return status

    def _get_values(self, document: Dict) -> List[any]: # type: ignore
        """Convert the chunk into a list of values to insert whose indexes align with fields.

        Args:
            chunk (DocumentChunk): The chunk to convert.

        Returns:
            List (any): The values to insert.
        """

        # create timestamp
        document["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # List to collect data we will return
        ret = []
        # Grab data responding to each field, excluding the hidden auto pk field for schema V1

        for key, _, _ in self._get_schema():
            # Grab the data at the key and default to our defaults set in init
            x = document.get(key)
            # Add the corresponding value if it passes the tests
            ret.append(x)
        return ret

    def _get_filter(self, query: Dict) -> Optional[str]:
        """Converts a text_ids to the expression that Milvus takes.

        Args:
            text_ids (List[Text]): a list of text_id.

        Returns:
            Optional[str]: The filter if valid, otherwise None.
        """
        # Join all our expressions with `and``
        
        filter_txts = []
        # if "user_ids" in query.keys() and not None is query['user_ids']:
        #     filter_txts.append(f"user_id in \"[{','.join(query['user_ids'])}]\"")
        # if "knowledge_ids" in query.keys() and not None is query['knowledge_ids']:
        #     filter_txts.append(f"knowledge_id in \"[{','.join(query['knowledge_ids'])}]\"")
        if "text_ids" in query.keys() and not None is query['text_ids']:
            text_ids_str = '", "'.join(query['text_ids'])
            filter_txts.append(f"text_id in [\"{text_ids_str}\"]")
        if "version" in query.keys():
            filter_txts.append(f"version == \"{query['version']}\"")
        return f"{' && '.join(filter_txts)}" or None


    def _get_output_fields(self) -> List[Text]:
        return [field[0] for field in self._get_schema() if field[0] != EMBEDDING_FIELD]

    @staticmethod
    def _is_collection_exists(collection_name: Text) -> bool:
        collection_list = utility.list_collections()
        return collection_name in collection_list

    def load_to_buffer(self, collection_name: Text, collection: Collection) -> None:
        """
        load milvus collection instance to buffer
        :param collection_name:
        :param collection:
        """
        if collection_name in self.buffer:
            logger.error(f"collection {collection_name} already in buffer, please check your code!!!")
        else:
            collection.load()
            self.buffer[collection_name] = collection

    def release_from_buffer(self, collection_name: Text) -> Optional[Collection]:
        """
        release milvus collection instance from buffer
        :param collection_name:
        :return:
        """
        if collection_name not in self.buffer:
            logger.error(f"collection {collection_name} not in buffer, please check your code!!!")
        else:
            collection = self.buffer.pop(collection_name)
            collection.release()
            return collection

    def drop(self, collection_name: Text) -> None:
        collection = self.buffer.get(collection_name, None)
        if collection is not None:
            _ = self.release_from_buffer(collection_name)
        utility.drop_collection(collection_name)


