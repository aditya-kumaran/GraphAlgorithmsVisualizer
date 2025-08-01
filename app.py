import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from graph_algorithms.floyd_warshall_algorithm import floyd_warshall_algorithm

try:
    from algorithms.bfs import bfs_algorithm
    from algorithms.dfs import dfs_algorithm
    from algorithms.dijkstra import dijkstra_algorithm
except ImportError:
    st.warning("Algorithm modules not found. Please implement the algorithms in the 'algorithms' directory.")
    bfs_algorithm = None
    dfs_algorithm = None
    dijkstra_algorithm = None


class GraphVisualizer:
    def __init__(self):
        self.graph = nx.Graph()
        self.pos = {}
        self.algorithm_results = {}

    def create_sample_graph(self, graph_type: str) -> nx.Graph:
        """Create different types of sample graphs"""
        if graph_type == "Grid":
            G = nx.grid_2d_graph(5, 5)
            G = nx.convert_node_labels_to_integers(G)
        elif graph_type == "Random":
            G = nx.erdos_renyi_graph(15, 0.3)
        elif graph_type == "Complete":
            G = nx.complete_graph(8)
        elif graph_type == "Tree":
            G = nx.balanced_tree(3, 3)
        else:  # Custom
            G = nx.Graph()

        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)

        return G

    def draw_graph(self, G: nx.Graph, highlighted_nodes: List = None,
                   highlighted_edges: List = None, path: List = None):
        """Draw the graph with optional highlighting"""
        fig, ax = plt.subplots(figsize=(12, 8))

        if not self.pos or len(self.pos) != len(G.nodes()):
            self.pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_edges(G, self.pos, edge_color='lightgray',
                               width=1, ax=ax)

        if highlighted_edges:
            nx.draw_networkx_edges(G, self.pos, edgelist=highlighted_edges,
                                   edge_color='red', width=3, ax=ax)

        if path and len(path) > 1:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G, self.pos, edgelist=path_edges,
                                   edge_color='blue', width=4, ax=ax)

        nx.draw_networkx_nodes(G, self.pos, node_color='lightblue',
                               node_size=500, ax=ax)

        if highlighted_nodes:
            nx.draw_networkx_nodes(G, self.pos, nodelist=highlighted_nodes,
                                   node_color='orange', node_size=600, ax=ax)

        if path:
            nx.draw_networkx_nodes(G, self.pos, nodelist=path,
                                   node_color='green', node_size=700, ax=ax)

        nx.draw_networkx_labels(G, self.pos, ax=ax)

        if nx.is_weighted(G):
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, self.pos, edge_labels, ax=ax)

        ax.set_title("Graph Visualization")
        ax.axis('off')
        plt.tight_layout()
        return fig


def main():
    st.set_page_config(page_title="Graph Algorithm Visualizer", layout="wide")

    st.title("🔍 Graph Algorithm Visualizer")
    st.markdown("Visualize BFS, DFS, Dijkstra, and Floyd-Warshall algorithms on different graph structures")

    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = GraphVisualizer()

    visualizer = st.session_state.visualizer

    with st.sidebar:
        st.header("Graph Configuration")

        graph_type = st.selectbox(
            "Select Graph Type:",
            ["Grid", "Random", "Complete", "Tree", "Custom"]
        )

        if st.button("Generate New Graph"):
            visualizer.graph = visualizer.create_sample_graph(graph_type)
            visualizer.pos = {}  # Reset positions
            st.rerun()

        st.divider()

        st.header("Algorithm Selection")
        algorithm = st.selectbox(
            "Choose Algorithm:",
            ["BFS (Breadth-First Search)", "DFS (Depth-First Search)", "Dijkstra's Algorithm", "Floyd-Warshall Algorithm"]
        )

        if len(visualizer.graph.nodes()) > 0:
            nodes = list(visualizer.graph.nodes())

            start_node = st.selectbox("Start Node:", nodes, index=0)

            if algorithm == "Dijkstra's Algorithm":
                end_node = st.selectbox("End Node:", nodes,
                                        index=min(1, len(nodes) - 1))
            else:
                end_node = None
        else:
            start_node = None
            end_node = None

        run_algorithm = st.button("Run Algorithm", type="primary")

        st.divider()

        if graph_type == "Custom":
            st.header("Custom Graph Builder")

            col1, col2 = st.columns(2)
            with col1:
                node_to_add = st.number_input("Add Node:", min_value=0, value=0)
                if st.button("Add Node"):
                    visualizer.graph.add_node(node_to_add)
                    st.rerun()

            with col2:
                if len(visualizer.graph.nodes()) >= 2:
                    nodes = list(visualizer.graph.nodes())
                    edge_start = st.selectbox("Edge Start:", nodes, key="edge_start")
                    edge_end = st.selectbox("Edge End:", nodes, key="edge_end")
                    edge_weight = st.number_input("Weight:", min_value=1, value=1)

                    if st.button("Add Edge"):
                        visualizer.graph.add_edge(edge_start, edge_end, weight=edge_weight)
                        st.rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Graph Visualization")

        if len(visualizer.graph.nodes()) == 0:
            st.info("Please generate a graph to start visualization")
            visualizer.graph = visualizer.create_sample_graph("Grid")

        if len(visualizer.graph.nodes()) > 0:
            fig = visualizer.draw_graph(visualizer.graph)
            st.pyplot(fig)
            plt.close()

        if run_algorithm and (start_node is not None or algorithm == "Floyd-Warshall Algorithm"):
            st.header("Algorithm Execution")

            if algorithm == "BFS (Breadth-First Search)":
                if bfs_algorithm:
                    with st.spinner("Running BFS..."):
                        result = bfs_algorithm(visualizer.graph, start_node)
                        st.success("BFS completed!")

                        if 'visited_order' in result:
                            st.write("**Visited nodes in order:**", result['visited_order'])
                        if 'tree_edges' in result:
                            fig = visualizer.draw_graph(
                                visualizer.graph,
                                highlighted_nodes=result.get('visited_order', []),
                                highlighted_edges=result.get('tree_edges', [])
                            )
                            st.pyplot(fig)
                            plt.close()
                else:
                    st.error("BFS algorithm not implemented. Please implement `bfs_algorithm` in `algorithms/bfs.py`")

            elif algorithm == "DFS (Depth-First Search)":
                if dfs_algorithm:
                    with st.spinner("Running DFS..."):
                        result = dfs_algorithm(visualizer.graph, start_node)
                        st.success("DFS completed!")

                        if 'visited_order' in result:
                            st.write("**Visited nodes in order:**", result['visited_order'])
                        if 'tree_edges' in result:
                            fig = visualizer.draw_graph(
                                visualizer.graph,
                                highlighted_nodes=result.get('visited_order', []),
                                highlighted_edges=result.get('tree_edges', [])
                            )
                            st.pyplot(fig)
                            plt.close()
                else:
                    st.error("DFS algorithm not implemented. Please implement `dfs_algorithm` in `algorithms/dfs.py`")

            elif algorithm == "Dijkstra's Algorithm":
                if dijkstra_algorithm and end_node is not None:
                    with st.spinner("Running Dijkstra's Algorithm..."):
                        result = dijkstra_algorithm(visualizer.graph, start_node, end_node)
                        st.success("Dijkstra's Algorithm completed!")

                        if 'shortest_path' in result:
                            st.write("**Shortest path:**", result['shortest_path'])
                            st.write("**Path length:**", result.get('path_length', 'N/A'))
                        if 'distances' in result:
                            st.write("**All distances from start node:**")
                            for node, dist in result['distances'].items():
                                st.write(f"Node {node}: {dist}")

                        if 'shortest_path' in result:
                            fig = visualizer.draw_graph(
                                visualizer.graph,
                                path=result['shortest_path']
                            )
                            st.pyplot(fig)
                            plt.close()
                else:
                    if not dijkstra_algorithm:
                        st.error("Dijkstra's algorithm not implemented. Please implement `dijkstra_algorithm` in `algorithms/dijkstra.py`")
                    else:
                        st.error("Please select both start and end nodes for Dijkstra's algorithm")

            elif algorithm == "Floyd-Warshall Algorithm":
                with st.spinner("Running Floyd-Warshall Algorithm..."):
                    result = floyd_warshall_algorithm(visualizer.graph)
                    st.success("Floyd-Warshall completed!")

                    distance = result['distance']
                    paths = result['shortest_paths']
                    infinite_pairs = result['infinite_pairs']

                    st.subheader("All-Pairs Shortest Paths")
                    for u in sorted(visualizer.graph.nodes()):
                        for v in sorted(visualizer.graph.nodes()):
                            if u == v:
                                continue
                            path = paths[u][v]
                            dist = distance[u][v]
                            if path:
                                st.write(f"**{u} → {v}**: Path = {path}, Distance = {dist}")
                            else:
                                st.write(f"**{u} → {v}**: ❌ No path")

                    st.subheader("Sample Shortest Path Visualization")
                    drawable_path = None
                    for u in sorted(paths):
                        for v in sorted(paths[u]):
                            if u != v and paths[u][v]:
                                drawable_path = paths[u][v]
                                break
                        if drawable_path:
                            break

                    if drawable_path:
                        fig = visualizer.draw_graph(
                            visualizer.graph,
                            path=drawable_path
                        )
                        st.pyplot(fig)
                        plt.close()

    with col2:
        st.header("Graph Information")

        if len(visualizer.graph.nodes()) > 0:
            st.metric("Number of Nodes", len(visualizer.graph.nodes()))
            st.metric("Number of Edges", len(visualizer.graph.edges()))
            st.metric("Graph Type", "Weighted" if nx.is_weighted(visualizer.graph) else "Unweighted")

            st.subheader("Adjacency List")
            for node in sorted(visualizer.graph.nodes()):
                neighbors = list(visualizer.graph.neighbors(node))
                if neighbors:
                    st.write(f"**Node {node}:** {neighbors}")
                else:
                    st.write(f"**Node {node}:** No connections")

        st.header("Algorithm Information")

        if algorithm == "BFS (Breadth-First Search)":
            st.markdown("""
            **Breadth-First Search (BFS)**
            - Explores nodes level by level
            - Uses a queue data structure
            - Finds shortest path in unweighted graphs
            - Time complexity: O(V + E)
            - Space complexity: O(V)
            """)

        elif algorithm == "DFS (Depth-First Search)":
            st.markdown("""
            **Depth-First Search (DFS)**
            - Explores as far as possible along each branch
            - Uses a stack data structure (or recursion)
            - Good for topological sorting, cycle detection
            - Time complexity: O(V + E)
            - Space complexity: O(V)
            """)

        elif algorithm == "Dijkstra's Algorithm":
            st.markdown("""
            **Dijkstra's Algorithm**
            - Finds shortest paths from source to all vertices
            - Works with non-negative edge weights
            - Uses a priority queue
            - Time complexity: O((V + E) log V)
            - Space complexity: O(V)
            """)

        elif algorithm == "Floyd-Warshall Algorithm":
            st.markdown("""
            **Floyd-Warshall Algorithm**
            - Computes shortest paths between all pairs of nodes
            - Works with weighted graphs (even with negative weights, but no negative cycles)
            - Time complexity: O(V³)
            - Space complexity: O(V²)
            """)


if __name__ == "__main__":
    main()
