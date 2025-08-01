# Graph Algorithm Visualizer

A Streamlit web application for visualizing graph algorithms including BFS (Breadth-First Search), DFS (Depth-First Search), and Dijkstra's Algorithm.

## Features

- **Interactive Graph Creation**: Generate different types of graphs (Grid, Random, Complete, Tree) or build custom graphs
- **Algorithm Visualization**: Visualize the execution of BFS, DFS, and Dijkstra's algorithms
- **Real-time Results**: See visited nodes, tree edges, shortest paths, and algorithm metrics
- **Modular Design**: Easy to extend with additional algorithms

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure

```
graph_visualizer/
├── app.py                 # Main Streamlit application
├── algorithms/            # Algorithm implementations
│   ├── __init__.py
│   ├── bfs.py            # BFS algorithm
│   ├── dfs.py            # DFS algorithm
│   └── dijkstra.py       # Dijkstra's algorithm
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Algorithm Modules

The application is designed with a modular structure where each algorithm is implemented in its own module:

### BFS Algorithm (`algorithms/bfs.py`)
- Function: `bfs_algorithm(graph, start_node)`
- Returns: Dictionary with `visited_order`, `tree_edges`, and `distances`

### DFS Algorithm (`algorithms/dfs.py`)
- Function: `dfs_algorithm(graph, start_node)`
- Returns: Dictionary with `visited_order`, `tree_edges`, `discovery_time`, and `finish_time`

### Dijkstra's Algorithm (`algorithms/dijkstra.py`)
- Function: `dijkstra_algorithm(graph, start_node, end_node)`
- Returns: Dictionary with `shortest_path`, `path_length`, `distances`, and `previous`

## Usage

1. **Select Graph Type**: Choose from predefined graph types or create a custom graph
2. **Generate Graph**: Click "Generate New Graph" to create a new graph structure
3. **Choose Algorithm**: Select BFS, DFS, or Dijkstra's algorithm
4. **Set Parameters**: Choose start node (and end node for Dijkstra's)
5. **Run Algorithm**: Click "Run Algorithm" to execute and visualize the results

## Customization

### Adding New Algorithms

1. Create a new Python file in the `algorithms/` directory
2. Implement your algorithm function following the same pattern
3. Import and integrate it into the main `app.py` file

### Modifying Visualizations

The `GraphVisualizer` class in `app.py` handles all graph drawing. You can customize:
- Node colors and sizes
- Edge colors and weights
- Layout algorithms
- Additional visual elements

## Graph Types

- **Grid**: 5x5 grid graph
- **Random**: Erdős–Rényi random graph
- **Complete**: Complete graph with all possible edges
- **Tree**: Balanced tree structure
- **Custom**: Build your own graph by adding nodes and edges

## Algorithm Information

### BFS (Breadth-First Search)
- Explores nodes level by level
- Time complexity: O(V + E)
- Space complexity: O(V)
- Finds shortest path in unweighted graphs

### DFS (Depth-First Search)
- Explores as far as possible along each branch
- Time complexity: O(V + E)
- Space complexity: O(V)
- Good for topological sorting and cycle detection

### Dijkstra's Algorithm
- Finds shortest paths with non-negative edge weights
- Time complexity: O((V + E) log V)
- Space complexity: O(V)
- Works on weighted graphs

## Contributing

Feel free to extend this application by:
- Adding new graph algorithms
- Improving visualizations
- Adding animation features
- Implementing step-by-step execution
- Adding more graph types

## License

This project is open source and available under the MIT License.
