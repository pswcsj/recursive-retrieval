import time
import os
import pickle
from typing import Any, Dict, List, Optional, cast

import numpy as np
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)

from typing import Literal, Union


DocType = Union[Literal["kifrs"], Literal["kgaap"]]

DEFAULT_PERSIST_ARGS = {"similarity_top_k": "similarity_top_k", "_verbose": "verbose"}

DEFAULT_PERSIST_FILENAME = "bm25_retriever.json"


class BM25Retriever(BaseRetriever):

    def __init__(
        self,
        doc_type: DocType,
        nodes: Optional[List[BaseNode]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        persist_dir: str = "storage",
        refresh: bool = False,
    ) -> None:
        self.similarity_top_k = similarity_top_k
        self.persist_dir = os.path.join(persist_dir, doc_type)
        self.persist_path = os.path.join(self.persist_dir, DEFAULT_PERSIST_FILENAME)
        if nodes is None:
            raise ValueError("Please pass nodes or an existing BM25 object.")

        if not refresh and os.path.exists(self.persist_path):
            self.load_index()
        else:
            if nodes is None:
                raise ValueError(
                    "Please pass nodes or set refresh=False with an existing persisted index."
                )
            self.build_index(nodes)

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def build_index(self, nodes: List[BaseNode]):
        self.corpus = [node_to_metadata_dict(node) for node in nodes]
        self.okt = Okt()
        print("Start indexing...")
        self.corpus_tokens = [self.okt.morphs(node.get_content()) for node in nodes]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("Finished indexing")
        self.persist_index()

    def persist_index(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(
                {
                    "corpus": self.corpus,
                    "corpus_tokens": self.corpus_tokens,
                    "bm25": self.bm25,
                },
                f,
            )
        print(f"BM25 index persisted to {self.persist_path}")

    def load_index(self):
        print(f"Loading BM25 index from {self.persist_path}")
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.corpus_tokens = data["corpus_tokens"]
        self.bm25 = data["bm25"]
        self.okt = Okt()
        print("BM25 index loaded successfully")

    @classmethod
    def from_defaults(
        cls,
        doc_type: DocType,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        persist_dir: str = "storage",
        refresh: bool = False,
        # deprecated
    ) -> "BM25Retriever":

        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            doc_type=doc_type,
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            persist_dir=persist_dir,
            refresh=refresh,
        )

    def get_persist_args(self) -> Dict[str, Any]:
        """Get Persist Args Dict to Save."""
        return {
            DEFAULT_PERSIST_ARGS[key]: getattr(self, key)
            for key in DEFAULT_PERSIST_ARGS
            if hasattr(self, key)
        }

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = self.okt.morphs(query)

        scores = self.bm25.get_scores(tokenized_query)
        # Filter out zero scores
        non_zero_indices = np.nonzero(scores)[0]
        non_zero_scores = scores[non_zero_indices]

        # Sort the non-zero scores in descending order
        sorted_indices = np.argsort(non_zero_scores)[::-1]

        # Select top k
        top_k_indices = sorted_indices[: self.similarity_top_k]

        # Get the original indices and scores
        selected_indices = non_zero_indices[top_k_indices]
        selected_scores = non_zero_scores[top_k_indices]

        nodes: List[NodeWithScore] = []
        for idx, score in zip(selected_indices, selected_scores):
            # idx can be an int or a dict of the node
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes
