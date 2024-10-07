from typing import List, Dict, Any, Optional, Literal, TypedDict

from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI as LLamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

class MultiAgentSearchLocalNode(TextNode):
    """
    An extended version of LlamaIndex's TextNode, with support for Element Types from Reducto
    Also support for chunk IDs from WhyHow
    Supports a tree-recursive print function for outputting structure as YAML for LLM calss
    """

    children: Dict[str, "MultiAgentSearchLocalNode"] = {}
    context: Optional[Dict[str, Any]] = {}

    def __init__(self, *, unique_id: str, content: str, **data: Any):
        print(f"Creating node with id {unique_id} {content}")
        super().__init__(text=content, id_=unique_id, **data)
 

    # Implement hash and eq for use in dictionaries and sets
    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultiAgentSearchLocalNode):
            return self.node_id == other.node_id
        return False

    # print node for prompt. use yaml to save tokens + better performance than json
    # recursive tree printing function
    # GPT will return with JSON
    def print_node_prompt(self, indent_level: int = 0, show_type=False) -> str:
        cleaned_text = " ".join(self.text.split())
        indent_unit = "  "
        indent = indent_unit * indent_level

        prompt = f"""{indent}- node_id: `{self.node_id}`"""
        
        if show_type:
            prompt += f"{indent}{indent_unit*2}type: {self.type}\n"

        prompt += f"{indent}{indent_unit*2}content: {cleaned_text}"
        
        if self.context.items():
            context = ""
            for key, value in self.context.items():
                if value not in [None, "", " "]:
                    context += f"{indent}{indent_unit*3}- {key}: {value}\n"
            prompt += f"{indent}{indent_unit*2}context:\n"
            prompt += f"{context}\n"
        children = list(self.children.values())
        if len(children) > 0:
            prompt += f"{indent}{indent_unit*2}children: \n"
            for child in children:
                prompt += f"{child.print_node_prompt(indent_level = indent_level + 5)}"
        return prompt
    
class AgentState(TypedDict):
    # we keep track of existing messages
    query: str
    definitions: Dict[str, str]
    previous_nodes: List[MultiAgentSearchLocalNode]
    last_fetched_context_nodes: List[MultiAgentSearchLocalNode]
    node_links_to_fetch: List[str]
    node_footers_to_fetch: Dict[str, str]
    allow_continue: bool
    search_failures: List[str]
    markdown_debug: str
    pass_count: int
    references: List[str]



