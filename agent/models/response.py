from typing import List
from pydantic import BaseModel


class FooterSearchResponse(BaseModel):
    """
    Pydantic schema used to contain the response for a node that needs a footer search, and the query involved.
    """

    node_id: str
    search_query: str


class RouterAgentFooterResponse(BaseModel):
    """
    Pydantic schema used to describe the structured output of the router agent.
    """

    node_ids_for_footer_search: List[FooterSearchResponse]
    
class RouterAgentLinkResponse(BaseModel):
    """
    Pydantic schema used to describe the structured output of the router agent.
    """

    node_ids_for_link_retrieval: List[str]

class PruneNodeIdResponse(BaseModel):
    """
    Pydantic schema used to describe a node via node_id being returned by the LLM.
    """

    node_id: str
    reason: str


class PruneNodeIdsResponse(BaseModel):
    """
    Pydantic schema used to describe nodes via node_ids being returned by the LLM.
    """

    content: List[PruneNodeIdResponse]