from typing import List, Any




local_nodes = list(local_nodes_map.values())
starting_nodes = local_nodes[100:120]
nodes_test3, change_count = fill_nodes_with_link_hints(starting_nodes)


for node in nodes_test3:
    print(node.print_node_prompt())

print(change_count)   