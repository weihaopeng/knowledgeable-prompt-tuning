"""Wrapper around the Milvus vector database."""
from __future__ import annotations
import time
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Text, Dict

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

from app.datastore.providers.milvus_datastore import MilvusDataStore


class MilvusStore(VectorStore):
    """Wrapper around the Milvus vector database."""

    def __init__(
        self,
        embedding_function: Embeddings,
        data_store: MilvusDataStore
    ):
        """Initialize wrapper around the milvus vector database.

        In order to use this you need to have `pymilvus` installed and a
        running Milvus instance.

        See the following documentation for how to run a Milvus instance:
        https://milvus.io/docs/install_standalone-docker.md

        Args:
            embedding_function (Embeddings): Function used to embed the text
            connection_args (dict): Arguments for pymilvus connections.connect()
            collection_name (str): The name of the collection to search.
            text_field (str): The field in Milvus schema where the
                original text is stored.
        """

        self.embedding_func = embedding_function
        self.data_store = data_store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        store_name: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]: # type: ignore
        """Insert text data into Milvus.

        When using add_texts() it is assumed that a collecton has already
        been made and indexed. If metadata is included, it is assumed that
        it is ordered correctly to match the schema provided to the Collection
        and that the embedding vector is the first schema field.

        Args:
            texts (Iterable[str]): The text being embedded and inserted.
            metadatas (Optional[List[dict]], optional): The metadata that
                corresponds to each insert. Defaults to None.
            partition_name (str, optional): The partition of the collection
                to insert data into. Defaults to None.
            timeout: specified timeout.

        Returns:
            List[str]: The resulting keys for each inserted element.
        """
        pass

    def _worker_search(
        self,
        query: Text,
        store_name: Text,
        text_ids: Optional[List[Text]] = None,
        k: int = 4,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[float], List[Tuple[Document, Any, Any]]]:
        # Organize results.
        ret = []

        query_embedding = self.embedding_func.embed_query(query)

        input = {"query": query, "embedding": query_embedding, "text_ids": text_ids, "top_k": k}
        query_results = self.data_store.query(store_name, queries=[input])

        if query_results is not None and len(query_results) > 0:
            query_results = query_results[0]['results']
            for result in query_results:
                ret.append(
                    (
                        Document(page_content=result["text"], metadata={"source": result["text"], "source_id": result["text_id"]}),
                        result["score"],
                        result["id"] if hasattr(result, "id") else None,
                    )
                )

        return query_embedding, ret

    def similarity_search_with_score(
        self,
        query: Text,
        text_ids: List[Text] = None,
        k: int = 4,
        store_name: Text = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results.

        Args:
            query (str): The text being searched.
            k (int, optional): The amount of results ot return. Defaults to 4.
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            partition_names (List[str], optional): Partitions to search through.
                Defaults to None.
            round_decimal (int, optional): Round the resulting distance. Defaults
                to -1.
            timeout (int, optional): Amount to wait before timeout error. Defaults
                to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[float], List[Tuple[Document, any, any]]: search_embedding,
                (Document, distance, primary_field) results.
        """
        _, result = self._worker_search(
            query, store_name, text_ids, k, timeout, **kwargs
        )
        return [(x, y) for x, y, _ in result]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            param (dict, optional): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            partition_names (List[str], optional): What partitions to search.
                Defaults to None.
            round_decimal (int, optional): Round the resulting distance. Defaults
                to -1.
            timeout (int, optional): Amount to wait before timeout error. Defaults
                to None.

        Returns:
            List[Document]: Document results for search.
        """
        data, res = self._worker_search(
            query,
            fetch_k,
            param,
            expr,
            partition_names,
            round_decimal,
            timeout,
            **kwargs,
        )
        # Extract result IDs.
        ids = [x for _, _, x in res]
        # Get the raw vectors from Milvus.
        vectors = self.col.query(
            expr=f"{self.primary_field} in {ids}",
            output_fields=[self.primary_field, self.vector_field],
        )
        # Reorganize the results from query to match result order.
        vectors = {x[self.primary_field]: x[self.vector_field] for x in vectors}
        search_embedding = data
        ordered_result_embeddings = [vectors[x] for x in ids]
        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(search_embedding), ordered_result_embeddings, k=k
        )
        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            if x == -1:
                break
            else:
                ret.append(res[x][0])
        return ret

    def similarity_search(
        self,
        query: Text,
        text_ids: List[Text] = None,
        k: int = 4,
        store_name: Text = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            text_ids (List[Text]): text_ids to filter search data
            k (int): recall top k docs
            store_name (Text): store hte
            timeout (int, optional): How long to wait before timeout error.
                Defaults to None.

        Returns:
            List[Document]: Document results for search.
        """
        _, docs_and_scores = self._worker_search(
            query, store_name, text_ids, k, timeout, **kwargs
        )
        return [doc for doc, _, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ):
        """Create a Milvus collection, indexes it with HNSW, and insert data.

        Args:
            texts (List[str]): Text to insert.
            embedding (Embeddings): Embedding function to use.
            metadatas (Optional[List[dict]], optional): Dict metatadata.
                Defaults to None.

        Returns:
            VectorStore: The Milvus vector store.
        """
        pass

    def create(self, store_name: Text):
        """
         create data storage
        :param store_name:
        """
        self.data_store.create(store_name)

    def upsert(self, store_name: Text, documents: List[Dict]):
        """
        :param store_name:
        :param documents: [
                {'id': '0_0', 'text_id': '0', 'text': '天王盖地虎'},
                {'id': '1_1', 'text_id': '1', 'text': '轻轻地放下这个杯子吧，老爹。'},
                {'id': '1_2', 'text_id': '1', 'text': 'a squat man is staggering out the arcade.'},
            ]
        """
        texts = [item["text"] for item in documents]
        embeddings = self.embedding_func.embed_documents(texts)
        assert len(texts) == len(embeddings)
        for i in range(len(texts)):
            documents[i]["embedding"] = embeddings[i]
        self.data_store.upsert(store_name, documents)

    def delete(self, store_name: Text, text_ids: List[Text] = None, delete_all: bool = False):
        """
        delete text by text ids in a data storage
        :param store_name:
        :param text_ids: text ids as filter
        :param delete_all: if set True, all text in the storage will be deleted
        """
        # TODO delete api
        self.data_store.delete(store_name, text_ids, delete_all)

