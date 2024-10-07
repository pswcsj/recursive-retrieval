# Define the State
import json
import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from agent.models.agent import AgentState, MultiAgentSearchLocalNode
from langgraph.graph import END, StateGraph, START

import tiktoken
from agent.models.response import (
    PruneNodeIdResponse,
    PruneNodeIdsResponse,
    RouterAgentLinkResponse,
)
from agent.models.agent import MultiAgentSearchLocalNode
from agent.utils import prune_node_from_nodes
from graph.lexical import LexicalGraphBuilder
from retriever.hybrid import VectorBM25
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

tokenizer = tiktoken.get_encoding("o200k_base")

TOKEN_LIMIT = 32000


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def count_tokens_for_nodes(nodes: List[MultiAgentSearchLocalNode]) -> int:
    return sum(count_tokens(node.print_node_prompt()) for node in nodes)


class Workflow:
    def __init__(
        self,
        lexical_graph: Optional[LexicalGraphBuilder] = None,
        retriever: Optional[BaseRetriever] = None,
    ):
        self.graph = StateGraph(AgentState)
        self._lexical_graph = (
            lexical_graph
            if lexical_graph
            else LexicalGraphBuilder(
                file_name="./kifrs_result.json", upload_new_csvs_on_build=True
            ).build_graph()
        )
        self._retriever: BaseRetriever = (
            retriever
            if retriever
            else VectorBM25(vector_top_k=5, bm25_top_k=20, mode="OR")
        )
        self.openai_client = self.setup_openai_client()
        self.local_nodes_map: Dict[str, MultiAgentSearchLocalNode] = {}

        for node, data in self._lexical_graph.nodes(data=True):
            print(f"Processing node: {node}")
            content = (
                data["content"]
                if "content" in data and data["content"] != ""
                else "(empty content)"
            )
            if len(content) > 10000:
                content = content[:10000] + "..."
            self.local_nodes_map[node] = MultiAgentSearchLocalNode(
                unique_id=node,
                content=content,
            )

    def retrieve(self, query: str) -> List[MultiAgentSearchLocalNode]:
        retrieved_nodes: List[NodeWithScore] = self._retriever.retrieve(
            QueryBundle(query_str=query)
        )
        retrieved_nodes_local_form: List[MultiAgentSearchLocalNode] = [
            self.local_nodes_map[node.node.node_id] for node in retrieved_nodes
        ]
        return retrieved_nodes_local_form

    def setup_openai_client(self):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        # Initialize OpenAI client
        openai_client = OpenAI()
        openai_client.api_key = openai_api_key
        return openai_client

    def fill_nodes_with_link_hints(
        self, current_nodes: List[MultiAgentSearchLocalNode]
    ) -> Tuple[List[MultiAgentSearchLocalNode], int]:
        """
        Given a tree/list of nodes, update the nodes with hints to the LLM if any of the nodes contain links.
        """

        # some nodes are nested, so recurse to get all nodes
        flattened_nodes: Dict[str, MultiAgentSearchLocalNode] = {}

        def recurse_index_nodes(current_nodes: List[MultiAgentSearchLocalNode]):
            for node in current_nodes:
                flattened_nodes[node.id_] = node
                if len(node.children) > 0:
                    recurse_index_nodes(list(node.children.values()))

        recurse_index_nodes(current_nodes)
        change_count = 0

        context_key = "link"
        for node in current_nodes:
            link_nodes = [
                tail
                for tail in self._lexical_graph.neighbors(node.id_)
                if self._lexical_graph[node.id_][tail]["type"] == "links_to"
            ]
            node.context[context_key] = link_nodes
            change_count += 1

        # recurse through the whole graph and add context markers inside any node with a link
        # show the llm hints that some clauses/text/etc link to external parts

        return current_nodes, change_count

    def prune_nodes(
        self, query: str, current_nodes: List[MultiAgentSearchLocalNode]
    ) -> List[MultiAgentSearchLocalNode]:
        """
        LLM call to remove any nodes that don't contain relevant information to the query.
        """

        starter_prompt = f"""
        당신은 문서에서 검색된 노드의 후처리 필터링을 담당하는 지능형 에이전트입니다. 노드는 다음 질의에 대한 답변을 제공해야 합니다: 
        ```{query}```
        
        아래는 자동으로 검색된 노드 목록입니다. 당신의 임무는 질의와 관련이 없는 노드를 식별하고, 그 노드가 제거되어야 하는 짧은 이유를 제시하는 것입니다.

        이 태스크는 recall이 매우 중요한 작업이니, 정말로 관련이 없는 것만 제거하고, 조금이라도 필요할 것 같은 정보는 절대 제거하지 마십시오.
        
        노드를 삭제하면 해당 노드 내의 모든 하위 노드도 함께 삭제된다는 점에 유의하십시오.

        노드는 node_id로 식별됩니다. (node_id, reason)의 목록을 반환하고, 이유는 해당 노드가 삭제되어야 하는 짧은 설명이어야 합니다. node_ids는 백틱(`)으로 묶어야 합니다.
        
        단계별로 신중하게 생각하여 삭제할 노드를 선택하십시오.
        """

        all_flattened_nodes: Dict[str, MultiAgentSearchLocalNode] = {}

        # node_id, (footer_text, further_explore_footer)

        def recurse_collect_nodes(current_nodes: List[MultiAgentSearchLocalNode]):
            """
            Recursively iterate via DFS on nodes, to index them by page number.
            """
            for node in current_nodes:
                all_flattened_nodes[node.id_] = node

                if len(node.children) > 0:
                    recurse_collect_nodes(list(node.children.values()))

        recurse_collect_nodes(current_nodes)

        printout = ""
        for node in all_flattened_nodes.values():
            printout += node.print_node_prompt()

        completion = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": starter_prompt},
                {"role": "user", "content": printout},
            ],
            response_format=PruneNodeIdsResponse,
        )

        node_id_response = completion.choices[0].message.parsed.content

        prune_dictionary_results: Dict[str, str] = {}
        if isinstance(node_id_response, list):
            for item in node_id_response:
                if isinstance(item, PruneNodeIdResponse):
                    if item.node_id[0] == "`":
                        node_id = item.node_id[1:-1]
                    else:
                        node_id = item.node_id
                    # print(f"Pruning node with node_id: {node_id} due to reason: {item.reason}")
                    prune_dictionary_results[node_id] = item.reason
                    prune_node_from_nodes(current_nodes, node_id)

        return current_nodes, prune_dictionary_results

    def inital_search_agent(self, state: AgentState) -> AgentState:
        state["markdown_debug"] += f"### Initial Search Agent\n"

        # vector_bm25_retriever = VectorBM25(vector_top_k=1, bm25_top_k=7, mode="OR")
        # def retrieve_with_vector_bm25(query, verbose = False):
        #     retrieved_nodes: NodeWithScore = vector_bm25_retriever.retrieve(
        #         QueryBundle(query_str=query)
        #     )
        #     retrieved_nodes_local_form: MultiAgentSearchLocalNode = [
        #         local_nodes_map[node.node.node_id] for node in retrieved_nodes
        #     ]
        #     if verbose:
        #         for node in retrieved_nodes:
        #             print(node)
        #     return retrieved_nodes_local_form

        # fetch nodes by vector and bm25 search
        initial_retrieved_nodes = self.retrieve(state["query"])
        print(len(initial_retrieved_nodes))

        # LLM prunes nodes that are not relevant
        after_pruning_nodes, _ = self.prune_nodes(
            state["query"], initial_retrieved_nodes
        )

        state[
            "markdown_debug"
        ] += f" - retrieved {len(initial_retrieved_nodes)} nodes, pruned to {len(after_pruning_nodes)} nodes\n"

        state["last_fetched_context_nodes"] = after_pruning_nodes
        return state

    def definition_agent(self, state: AgentState) -> AgentState:
        # fetch definitions
        state["markdown_debug"] += f"### Definitions Agent\n"
        print(f"query: {state['query']}")
        definitions_dict = definitions_search(state["query"])
        print(f"dfns: {definitions_dict}")
        state["definitions"] = definitions_dict

        state[
            "markdown_debug"
        ] += f"Retrieved {len(definitions_dict)} definitions for: \n"
        for term, definition in definitions_dict.items():
            state["markdown_debug"] += f"  - **{term}** : {definition}\n"
        return state

    def context_fetch_tool(self, state: AgentState) -> AgentState:
        state["markdown_debug"] += f"### Context Fetched for nodes:\n"
        current_nodes = state["last_fetched_context_nodes"]

        # mark links with nodes
        full_clause_nodes_with_link_hints, change_count = (
            self.fill_nodes_with_link_hints(current_nodes)
        )

        state["last_fetched_context_nodes"] = full_clause_nodes_with_link_hints
        state[
            "markdown_debug"
        ] += f"- fetched full clauses, added to {len(full_clause_nodes_with_link_hints)} nodes. {change_count} new links\n"

        return state

    def supervisor_agent(self, state: AgentState) -> AgentState:
        state["markdown_debug"] += f"### Supervisor Agent\n"
        # Look for search failures. This might be an instance where multiple searches were made for certain parts of the document, but no relevant information was found.
        # This means that the search has to be ended prematurely to prevent infinite loops.
        printout = ""
        for node in state["previous_nodes"]:
            printout += node.print_node_prompt()
        for node in state["last_fetched_context_nodes"]:
            printout += node.print_node_prompt()

        prompt = f""" You are a supervisor agent overseeing the multi-agent retrieval process of graph nodes from a document. The nodes are to answer the query: {state['query']}

    Below is a list of nodes that were automatically retrieved. If you are absolutely certain that the query can be fully and sufficiently answered with the current information alone and there is no need to check any further linked nodes, return END. Otherwise, return CONTINUE.

    Return only a single word, either END or CONTINUE. """
        state["markdown_debug"] += f"- **supervisor**: {printout}\n"
        completion = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": printout},
                {"role": "user", "content": str(state["search_failures"])},
            ],
        )

        response: str = completion.choices[0].message.content
        if "end" in response.lower():
            state["allow_continue"] = False
            return state
        elif "continue" in response.lower():
            state["allow_continue"] = True

        state["markdown_debug"] += f"- **supervisor**: continue\n"

        # check if requests will be in token limit
        # chatgpt is 32k, use 32k as baseline
        try:
            tokens = count_tokens_for_nodes(
                state["previous_nodes"]
            ) + count_tokens_for_nodes(state["last_fetched_context_nodes"])
        except:
            tokens = TOKEN_LIMIT + 1
        if tokens > TOKEN_LIMIT:
            state["markdown_debug"] += f"- warning: tokens over recommended limit\n"
            previous_nodes, changes_1 = self.prune_nodes(
                state["query"], state["previous_nodes"]
            )
            last_fetched_context_nodes, changes_2 = self.prune_nodes(
                state["query"], state["last_fetched_context_nodes"]
            )
            new_tokens = count_tokens_for_nodes(state["previous_nodes"]) + count_tokens(
                state["last_fetched_context_nodes"]
            )

            if new_tokens > TOKEN_LIMIT:
                # raise ValueError("Too many tokens")
                print("Warning: Tried pruning, but still too many tokens")
                state[
                    "markdown_debug"
                ] += f"- Warning: Tried pruning, but still too many tokens\n"
            if new_tokens < tokens:
                state["previous_nodes"] = previous_nodes
                state["last_fetched_context_nodes"] = last_fetched_context_nodes

                for node_id, reason in changes_1.items():
                    state[
                        "markdown_debug"
                    ] += f"- pruned node {node_id} due to {reason}\n"

                for node_id, reason in changes_2.items():
                    state[
                        "markdown_debug"
                    ] += f"- pruned node {node_id} due to {reason}\n"

        return state

    def router_agent(self, state: AgentState) -> AgentState:
        # decide if process should should stop or continue
        state["markdown_debug"] += f"### Router Agent\n"

        starter_prompt_link = f"""
            You are an intelligent agent overseeing a multi-agent retrieval process of graph nodes from a document. These nodes are to answer the query: 
            ```{state['query']}```
            
            Below this request is a list of nodes that were automatically retrieved. 
            
            If there are linked nodes and you are sure that the current nodes are enough to answer the query, return an empty response.

            If there isn't enough information, you must identify any linked nodes that could be worth exploring.
            
            If there are no further nodes worth analysing, return an empty response.
            
            Return a list of node_ids. ONLY RETURN NODE_IDS for NODES THAT ARE RELEVANT TO ANSWERING THE QUERY. Nodes are identified by node_id and must be quoted in backticks.
        """

        # collect latest nodes, and all nodes
        last_fetched_nodes_flattened: Dict[str, MultiAgentSearchLocalNode] = {}
        all_nodes_flattened: Dict[str, MultiAgentSearchLocalNode] = {}

        def recurse_collect_last_fetched_nodes(
            current_nodes: List[MultiAgentSearchLocalNode],
        ):
            """
            Recursively iterate via DFS on nodes, to index them by page number.
            """
            for node in current_nodes:
                last_fetched_nodes_flattened[node.id_] = node
                all_nodes_flattened[node.id_] = node

                if len(node.children) > 0:
                    recurse_collect_last_fetched_nodes(list(node.children.values()))

        recurse_collect_last_fetched_nodes(state["last_fetched_context_nodes"])

        def recurse_collect_all_nodes(current_nodes: List[MultiAgentSearchLocalNode]):
            """
            Recursively iterate via DFS on nodes, to index them by page number.
            """
            for node in current_nodes:
                all_nodes_flattened[node.id_] = node

                if len(node.children) > 0:
                    recurse_collect_all_nodes(list(node.children.values()))

        recurse_collect_all_nodes(
            state["previous_nodes"] + state["last_fetched_context_nodes"]
        )

        printout = ""
        for node in last_fetched_nodes_flattened.values():
            printout += node.print_node_prompt()

        completion = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": starter_prompt_link},
                {"role": "user", "content": printout},
            ],
            response_format=RouterAgentLinkResponse,
        )

        # Parse the response content
        explore_response = json.loads(completion.choices[0].message.content)
        state["markdown_debug"] += f"explore_response: {explore_response}\n"
        print(f"explore_response: {explore_response}")

        # Extract node_ids_for_link_retrieval
        # Ensure that the node_ids are not already in the all_nodes_flattened
        node_ids_for_link_retrieval = []
        for node_id in explore_response["node_ids_for_link_retrieval"]:
            # sometimes GPT returns node ids with or without backticks
            if "`" in node_id:
                node_id = node_id.replace("`", "")
            if node_id not in self._lexical_graph:
                continue
            for child in self._lexical_graph.neighbors(node_id):
                if self._lexical_graph[node_id][child].get("type") == "links_to":
                    if all_nodes_flattened.get(child) is None:
                        node_ids_for_link_retrieval.append(child)
            # if all_nodes_flattened.get(node_id) is None:
            #     node_ids_for_link_retrieval.append(node_id)

        # Update the state with the node_ids_for_link_retrieval and node_ids_for_footer_search
        state["node_links_to_fetch"] = node_ids_for_link_retrieval

        # debug prints
        state[
            "markdown_debug"
        ] += f"- found {len(node_ids_for_link_retrieval)} nodes to fetch links for\n"
        for node_id in node_ids_for_link_retrieval:
            state["markdown_debug"] += f"    - {node_id}\n"

        return state

    def answering_agent(self, state: AgentState) -> AgentState:
        state["markdown_debug"] += f"### Answering Agent\n"
        # answer the query
        prompt = f"""
    You are an answering agent. You will be given a list of document nodes that were automatically retrieved by the system. These nodes are to answer the query:
    ```{state['query']}```

    Give references to sections/paragraphs if possible, but do not output full node ids with backticks and the hash. 
        """

        printout = ""
        for node in state["previous_nodes"]:
            printout += node.print_node_prompt()

        for node in state["last_fetched_context_nodes"]:
            printout += node.print_node_prompt()

        completion = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": printout},
            ],
        )

        message = completion.choices[0].message.content
        print(f"response: {message}")

        state["markdown_debug"] += f"## Answer to Query: \n{message}\n"
        state["markdown_debug"] += f"## Source Nodes: \n```\n{printout}\n```"

        return state

    def recursive_retrieval(self, state: AgentState) -> AgentState:
        state["markdown_debug"] += f"### Recursive Retrieval\n"

        current_nodes = state["last_fetched_context_nodes"]

        for current_node in current_nodes:
            state["previous_nodes"].append(current_node)

        new_current_nodes = []

        # look up the nodes to fetch by id
        state[
            "markdown_debug"
        ] += f"{len(state['node_links_to_fetch'])} nodes with link(s)\n"

        for node_id in state["node_links_to_fetch"]:
            # sometimes GPT returns node ids with or without backticks
            if node_id[0] == "`":
                node_id = node_id[1:-1]
            if node_id in self.local_nodes_map:
                new_current_nodes.append(self.local_nodes_map[node_id])
                state["markdown_debug"] += f"  - {node_id}\n"
            else:
                state["search_failures"].append(
                    f"Failed to fetch node with id: {node_id}"
                )

        state[
            "markdown_debug"
        ] += f"{len(state['node_footers_to_fetch'])} nodes with footer info\n"

        state["last_fetched_context_nodes"] = new_current_nodes
        state["pass_count"] += 1
        state["markdown_debug"] += f"# PASS {state['pass_count']}\n"
        state["node_footers_to_fetch"] = {}
        state["node_links_to_fetch"] = []

        return state

    def setup_workflow(self):
        self.graph.add_node("Initial Search Agent", self.inital_search_agent)
        self.graph.add_node("Context Fetch Tool", self.context_fetch_tool)
        self.graph.add_node("Supervisor Agent", self.supervisor_agent)
        self.graph.add_node("Router Agent", self.router_agent)
        self.graph.add_node("Answering Agent", self.answering_agent)
        self.graph.add_node("Recursive Retrieval", self.recursive_retrieval)

        self.graph.add_edge(START, "Initial Search Agent")
        self.graph.add_edge("Initial Search Agent", "Context Fetch Tool")
        self.graph.add_edge("Context Fetch Tool", "Supervisor Agent")
        self.graph.add_conditional_edges(
            "Supervisor Agent",
            self.should_continue_to_router,
            {
                True: "Router Agent",
                False: "Answering Agent",
            },
        )
        self.graph.add_conditional_edges(
            "Router Agent",
            self.should_continue_to_recursive_retrieval,
            {
                True: "Recursive Retrieval",
                False: "Answering Agent",
            },
        )
        self.graph.add_edge("Recursive Retrieval", "Context Fetch Tool")
        self.graph.add_edge("Answering Agent", END)

    @staticmethod
    def should_continue_to_router(state: AgentState) -> bool:
        return state["allow_continue"]

    @staticmethod
    def should_continue_to_recursive_retrieval(state: AgentState) -> bool:
        return len(state["node_links_to_fetch"]) > 0

    def run(self):
        self.setup_workflow()
        return self.graph.compile()
