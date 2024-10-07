from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import networkx as nx
from enum import Enum
import json


class LexicalGraphBuilder(BaseModel):

    file_name: str
    build_and_upload_csvs: bool = True

    documents: Optional[Dict[str, Any]] = None

    graph: Any = nx.DiGraph()
    kifrs_indices: List[int] = [
        1000,
        1001,
        1002,
        1007,
        1008,
        1010,
        1012,
        1016,
        1019,
        1020,
        1021,
        1023,
        1024,
        1026,
        1027,
        1028,
        1029,
        1032,
        1033,
        1034,
        1036,
        1037,
        1038,
        1039,
        1040,
        1041,
        1101,
        1102,
        1103,
        1105,
        1106,
        1107,
        1108,
        1109,
        1110,
        1111,
        1112,
        1113,
        1114,
        1115,
        1116,
        1117,
    ]

    def __init__(self, **data):
        super().__init__(**data)

        # Data structures to store Elements
        with open(self.file_name, "r") as file:
            self.documents = json.load(file)

    def create_new_node(self, node_id: str, type, content: str):
        self.graph.add_node(node_id, type=type, content=content)

    def link_nodes(self, head_id, tail_id, type):
        self.graph.add_edge(head_id, tail_id, type=type)

    def load_nodes(self):
        for unique_id, document in self.documents.items():
            doc_type = document["type"] if "type" in document else "document"
            content = (
                document["fullContent"]
                if doc_type == "paragraph"
                else document["title"]
            )
            self.create_new_node(unique_id, doc_type, content)

    def build_links_by_node(self, unique_id: str):
        document = self.documents[unique_id]

        doc_type = document["type"] if "type" in document else "document"

        if doc_type == "paragraph":
            parent_id = document["parents"][-1]
            self.link_nodes(parent_id, unique_id, "is_parent")
            explicit_links = document["explicit_links"]
            for link in explicit_links:
                self.link_nodes(unique_id, link, "links_to")
        elif doc_type == "title":
            parent_id = document["parents"][-1]
            self.link_nodes(parent_id, unique_id, "is_parent")

        children = document["children"]
        for child in children:
            self.build_links_by_node(child)

    def build_graph(self):
        self.load_nodes()

        for index_id in self.kifrs_indices:
            self.build_links_by_node(str(index_id))

        return self.graph
