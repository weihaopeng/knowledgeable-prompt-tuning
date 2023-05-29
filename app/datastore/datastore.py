from abc import ABC, abstractmethod
from typing import Text, Dict, List, Optional


class DataStore(ABC):

    @abstractmethod
    def create(self, store_name: Text) -> bool:
        """
        create a vector data store.
        Return create status
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self, store_name: Text, documents: List[Dict]
    ) -> bool:
        """
        Takes in a list of documents and inserts them into the database.
        First deletes all the existing vectors with the document id (if necessary, depends on the vector db), then inserts the new ones.
        Return a list of document ids.
            [
                {'id': '0_0', 'embedding': [0.2233, 0.1234, 0.0156 ..., 0.0890],  'text_id': '0', 'text': '天王盖地虎'},
                {'id': '1_1', 'embedding': [0.6729, 0.0211, -0.0529 ..., 0.0984], 'text_id': '1', 'text': '轻轻地放下这个杯子吧，老爹。'},
                {'id': '1_2', 'embedding': [-0.0023, 0.0372, 0.0333 ..., 0.0741], 'text_id': '1', 'text': 'a squat man is staggering out the arcade.'},
            ]
        """
        raise NotImplementedError

    def query(self, store_name: Text, queries: List[Dict]) -> Optional[List[Dict]]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        [
            {'query': '上山打老虎', 'embedding': [0.4653, 0.2334, 0.0290..., 0.1890], 'text_ids': None, 'top_k': 3},
            {'query': 'a man is walking through a big pipe', 'embedding': [0.2323, 0.9892, 0.1529..., -0.0614], 'text_ids': ['2', '5', '7'], 'top_k': 3},
            {'query': '轻轻地来，又轻轻地走', 'embedding': [0.1023, -0.0372, -0.8223..., 0.0092], 'text_id': None, 'top_k': 3},
        ]
        """

        return self._query(store_name, queries)

    @abstractmethod
    def _query(self, store_name: Text, queries: List[Dict]) -> Optional[List[Dict]]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        store_name: Text,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids in the datastore.
        Returns whether the operation was successful.
        """
        raise NotImplementedError

    @abstractmethod
    def drop(
        self,
        store_name: Text,
    ) -> None:
        """
        drop a datastore by store_name
        :param store_name:
        """
        raise NotImplementedError

