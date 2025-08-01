"""
Breadth-First Search (BFS) Algorithm Implementation

This module should contain the BFS algorithm implementation.
The function should return a dictionary with visualization data.
"""

import networkx as nx
from collections import deque
from typing import Dict, List, Any

def bfs_algorithm(graph: nx.Graph, start_node: int) -> Dict[str, Any]:
    """
    Perform Breadth-First Search on the given graph starting from start_node.
    
    Args:
        graph: NetworkX graph object
        start_node: Starting node for BFS
    
    Returns:
        Dictionary containing:
        - 'visited_order': List of nodes in the order they were visited
        - 'tree_edges': List of edges that form the BFS tree
        - 'distances': Dictionary of distances from start_node to each node
    
    Example implementation (replace with your own):
    """
    
    visited = set()
    visited_order = []
    tree_edges = []
    distances = {start_node: 0}
    queue = deque([start_node])
    
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            visited_order.append(current)
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
                    tree_edges.append((current, neighbor))
                    distances[neighbor] = distances[current] + 1
    
    return {
        'visited_order': visited_order,
        'tree_edges': tree_edges,
        'distances': distances
    }
