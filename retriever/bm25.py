import json
import pickle
import time

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
import os

DEFAULT_PERSIST_ARGS = {"similarity_top_k": "similarity_top_k", "_verbose": "verbose"}

DEFAULT_PERSIST_FILENAME = "retriever.json"


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        self.similarity_top_k = similarity_top_k
        self.verbose = verbose
        self.okt = Okt()

        if nodes is None:
            raise ValueError("Please pass nodes or use load_from_file=True.")

        self.corpus = [node_to_metadata_dict(node) for node in nodes]
        print("start indexing...")
        start_time = time.time()
        self.corpus_tokens = [self.okt.morphs(node.get_content()) for node in nodes]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"finished indexing within {time.time() - start_time} seconds")

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
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
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def persist(self, directory: str):
        """Persist the BM25Retriever object to a directory."""
        os.makedirs(directory, exist_ok=True)

        # Save simple data as JSON
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(
                {"similarity_top_k": self.similarity_top_k, "verbose": self.verbose}, f
            )

        # Save corpus as JSON
        with open(os.path.join(directory, "corpus.json"), "w") as f:
            json.dump(self.corpus, f)

        # Save corpus_tokens using pickle
        with open(os.path.join(directory, "corpus_tokens.pkl"), "wb") as f:
            pickle.dump(self.corpus_tokens, f)

        # Save BM25Okapi object using pickle
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)

        if self.verbose:
            print(f"BM25Retriever persisted to {directory}")

    @classmethod
    def load(cls, directory: str):
        """Load a BM25Retriever object from a directory."""
        # Load metadata
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Load corpus
        with open(os.path.join(directory, "corpus.json"), "r") as f:
            corpus = json.load(f)

        # Load corpus_tokens
        with open(os.path.join(directory, "corpus_tokens.pkl"), "rb") as f:
            corpus_tokens = pickle.load(f)

        # Load BM25Okapi object
        with open(os.path.join(directory, "bm25.pkl"), "rb") as f:
            bm25 = pickle.load(f)

        # Create nodes from corpus
        nodes = [metadata_dict_to_node(node_dict) for node_dict in corpus]

        # Create instance
        instance = cls(
            nodes=nodes,
            similarity_top_k=metadata["similarity_top_k"],
            verbose=metadata["verbose"],
        )

        # Set loaded attributes
        instance.corpus = corpus
        instance.corpus_tokens = corpus_tokens
        instance.bm25 = bm25

        if instance.verbose:
            print(f"BM25Retriever loaded from {directory}")

        return instance

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
