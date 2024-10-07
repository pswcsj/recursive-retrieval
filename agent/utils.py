from typing import Any, Dict, List, Tuple
from agent.models.agent import AgentState, MultiAgentSearchLocalNode


def prune_node_from_nodes(
    nodes: List[MultiAgentSearchLocalNode],
    node_id: str,
) -> None:
    """Prune a node tree by node_id."""

    def prune(node: MultiAgentSearchLocalNode) -> bool:
        if node.node_id == node_id:
            return True
        to_delete = []
        for child_id, child_node in node.children.items():
            if prune(child_node):
                to_delete.append(child_id)
        for child_id in to_delete:
            del node.children[child_id]
        return False

    nodes[:] = [root_node for root_node in nodes if not prune(root_node)]


def recursive_update(
    nodes: List[MultiAgentSearchLocalNode],
    node_id: str,
    property: str,
    property_value: Any,
) -> None:
    """Recursively search a node tree by node_id and update the specified property with the given value."""

    def update_node(node: MultiAgentSearchLocalNode) -> bool:
        if node.node_id == node_id:
            setattr(node, property, property_value)
            return True
        for child in node.children.values():
            if update_node(child):
                return True
        return False

    for root_node in nodes:
        if update_node(root_node):
            break


def recursive_add_context(
    nodes: List[MultiAgentSearchLocalNode],
    node_id: str,
    context_key: str,
    context_value: str,
) -> None:
    """Recursively search a node tree by node_id and add a key-value pair to the context dictionary of the matching node."""

    def add_context_to_node(node: MultiAgentSearchLocalNode) -> bool:
        if node.node_id == node_id:
            node.context[context_key] = context_value
            return True
        for child in node.children.values():
            if add_context_to_node(child):
                return True
        return False

    for root_node in nodes:
        if add_context_to_node(root_node):
            break

    return nodes


def recursive_append_context_items(
    nodes: List[MultiAgentSearchLocalNode],
    node_id: str,
    context_key: str,
    new_items: List[str],
) -> None:
    """Recursively search a node tree by node_id and add a key-value pair to the context dictionary of the matching node."""

    def add_context_link_to_node(node: MultiAgentSearchLocalNode) -> bool:
        if node.node_id == node_id:
            node_links = node.context.get(context_key, [])
            node_links.append(new_items)
            node.context[context_key] = node_links

            return True
        for child in node.children.values():
            if add_context_link_to_node(child):
                return True
        return False

    for root_node in nodes:
        if add_context_link_to_node(root_node):
            break

    return nodes


def recursive_check_if_exists(
    nodes: List[MultiAgentSearchLocalNode],
    node_id: str,
) -> bool:
    """Recursively search a node tree by node_id and check if the node exists."""

    def check_if_exists(node: MultiAgentSearchLocalNode) -> bool:
        if node.node_id == node_id:
            return True
        for child in node.children.values():
            if check_if_exists(child):
                return True
        return False

    for root_node in nodes:
        if check_if_exists(root_node):
            return True

    return False
