from typing import List
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from agent.models.agent import MultiAgentSearchLocalNode
from retriever.bm25 import BM25Retriever
from llama_index.core.schema import QueryBundle, NodeWithScore, BaseNode


class VectorBM25(BaseRetriever):
    """Custom retriever that performs both semantic search and BM25 search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        mode: str = "OR",
    ) -> "VectorBM25":
        """Create instance from both retrievers."""
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[MultiAgentSearchLocalNode]:
        """Retrieve nodes given query."""

        vector_nodes: List[NodeWithScore] = self._vector_retriever.retrieve(
            query_bundle
        )
        bm25_nodes: List[NodeWithScore] = self._bm25_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in bm25_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in bm25_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes
