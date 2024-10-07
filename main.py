from typing import Dict
from agent.main import Workflow
from agent.models.agent import AgentState, MultiAgentSearchLocalNode
from graph.lexical import LexicalGraphBuilder
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, StorageContext

from retriever.bm25 import BM25Retriever
from retriever.hybrid import VectorBM25
from llama_index.embeddings.openai import OpenAIEmbedding

debug_markdown = "# Multi-Agent Log\n"
debug_markdown += "## Pass 1\n"


def create_local_nodes_map(lexical_graph) -> Dict[str, MultiAgentSearchLocalNode]:
    local_nodes_map: Dict[str, MultiAgentSearchLocalNode] = {}
    for node, data in lexical_graph.nodes(data=True):
        print(f"Processing node: {node}")
        content = (
            data["content"]
            if "content" in data and data["content"] != ""
            else "(empty content)"
        )
        if len(content) > 10000:
            content = content[:10000] + "..."
        local_nodes_map[node] = MultiAgentSearchLocalNode(
            unique_id=node,
            content=content,
        )
    return local_nodes_map


def multi_agent_query(query="How can the Board and the CCO manage control functions?"):
    initial_state = AgentState(
        query=query,
        definitions={},
        previous_nodes=[],
        last_fetched_context_nodes=[],
        node_links_to_fetch=[],
        node_footers_to_fetch={},
        allow_continue=True,
        search_failures=[],
        markdown_debug=debug_markdown,
        pass_count=1,
    )
    lexical_graph = LexicalGraphBuilder(
        file_name="./kifrs_result.json", upload_new_csvs_on_build=True
    ).build_graph()
    local_nodes_map = create_local_nodes_map(lexical_graph)

    storage_context = StorageContext.from_defaults()
    openai_embeddings = OpenAIEmbedding(model="text-embedding-3-large")
    vector_index = VectorStoreIndex(
        nodes=list(local_nodes_map.values()),
        storage_context=storage_context,
        embed_model=openai_embeddings,
    )
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    bm25_retriever = BM25Retriever(
        nodes=list(local_nodes_map.values()), similarity_top_k=20
    )

    retriever = VectorBM25(
        vector_retriever=vector_retriever, bm25_retriever=bm25_retriever
    )
    workflow = Workflow(lexical_graph=lexical_graph, retriever=retriever)
    graph = workflow.run()
    events = graph.stream(initial_state, {"recursion_limit": 150})
    for s in events:
        first_value = next(iter(s.values()))
    # return all of agent's references after answer
    return list(map(lambda node: node.id_, first_value["previous_nodes"])) + list(
        map(lambda node: node.id_, first_value["last_fetched_context_nodes"])
    )


multi_agent_query(
    "회사는 선박 건조업을 영위하는 회사이다. 기능통화와 표시통화는 원화이며, 외화로 선박 건조 계약을 체결하였다. 해당 계약은 기업회계기준서 제1115호 ‘고객과의 계약에서 생기는 수익’ 문단 35(3)[1]을 충족하므로 수행의무는 기간에 걸쳐 이행되며, 이 거래에 비현금 대가는 존재하지 않는다고 가정한다. 회사는 기간에 걸쳐 수익을 인식할 때 발생 원가에 기초한 투입법으로 진행률을 계산하였다. (질의 1) 기간에 걸쳐 진행률에 따라 수익을 인식할 때, 관련 외화 계약자산을 최초 인식 시 적용해야 하는 환율은? (질의 2) 기간에 걸쳐 진행률에 따라 수익을 인식할 때, 관련 외화 계약자산은 화폐성자산인가?"
)
