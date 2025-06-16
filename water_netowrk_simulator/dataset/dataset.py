from pydantic import BaseModel, ConfigDict
from functools import cached_property
import polars as pl
from wntr.network.elements import Junction, Tank, Reservoir
import torch
import networkx as nx
import numpy as np

from alpha_leak_data.data import AlphaLeakData

from water_netowrk_simulator.enums import PipeMaterial
from water_netowrk_simulator.leak_simulator import LeakSimulator

from ..models import Node, Pipe


class LargeNetworkError(Exception):
    """
    Exception raised when attempting to compute edge paths for networks that are too large.
    
    Edge path computation has O(n²) complexity and becomes computationally expensive
    for large networks. This error is raised to prevent performance issues.
    """
    pass


class EdgePaths:
    """
    Efficient storage and retrieval of edge paths between nodes using NetworkX shortest paths.
    
    Computes shortest paths using NetworkX and stores them as sequences of edge indices.
    """
    
    def __init__(self, graph: nx.Graph, node_names: list[str], edge_names: list[tuple[str, str]], max_nodes: int = 50):
        """
        Initialize EdgePaths and compute all shortest paths.
        
        Parameters:
        -----------
        graph : nx.Graph
            NetworkX graph representing the network topology
        node_names : list[str]
            List of node names in order (matches node indices)
        edge_names : list[tuple[str, str]]
            List of edge names as (from_node, to_node) tuples (matches edge indices)
        max_nodes : int, default=50
            Maximum number of nodes allowed before raising LargeNetworkError
        """
        self.graph = graph
        self.node_names = node_names
        self.edge_names = edge_names
        self.node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(edge_names)}
        
        # Check network size to prevent performance issues
        if len(node_names) > max_nodes:
            raise LargeNetworkError(
                f"Network has {len(node_names)} nodes. Edge path computation is only "
                f"supported for networks with {max_nodes} or fewer nodes due to O(n²) complexity. "
                f"For larger networks, consider using distance matrix only."
            )
        
        # Sparse storage: {(from_node_idx, to_node_idx): [edge_indices]}
        self.paths: dict[tuple[int, int], list[int]] = {}
        
        # Compute all shortest paths
        self._compute_all_shortest_paths()
        
    def _compute_all_shortest_paths(self) -> None:
        """
        Compute shortest paths between all node pairs using NetworkX.
        """
        try:
            # Get all shortest paths as node sequences
            all_paths = dict(nx.all_pairs_shortest_path(self.graph))
            
            # Convert node paths to edge paths
            for from_node, paths_from_node in all_paths.items():
                from_idx = self.node_to_idx[from_node]
                
                for to_node, node_path in paths_from_node.items():
                    to_idx = self.node_to_idx[to_node]
                    
                    # Skip self-loops (same node)
                    if from_idx == to_idx:
                        self.paths[(from_idx, to_idx)] = []
                        continue
                    
                    # Convert node sequence to edge sequence
                    edge_indices = self._node_path_to_edge_path(node_path)
                    self.paths[(from_idx, to_idx)] = edge_indices
                    
        except nx.NetworkXError:
            # Handle disconnected graph
            for i, from_node in enumerate(self.node_names):
                for j, to_node in enumerate(self.node_names):
                    if i != j and (i, j) not in self.paths:
                        # Try to find path between these specific nodes
                        try:
                            node_path = nx.shortest_path(self.graph, from_node, to_node)
                            if isinstance(node_path, list):
                                edge_indices = self._node_path_to_edge_path(node_path)
                                self.paths[(i, j)] = edge_indices
                        except nx.NetworkXNoPath:
                            # No path exists - will be handled as missing key
                            pass
    
    def _node_path_to_edge_path(self, node_path: list[str]) -> list[int]:
        """
        Convert a sequence of nodes to a sequence of edge indices.
        
        Parameters:
        -----------
        node_path : list[str]
            Sequence of node names representing a path
            
        Returns:
        --------
        list[int]
            Sequence of edge indices
        """
        edge_indices = []
        
        for i in range(len(node_path) - 1):
            from_node = node_path[i]
            to_node = node_path[i + 1]
            
            # Try both directions for the edge
            edge = (from_node, to_node)
            reverse_edge = (to_node, from_node)
            
            if edge in self.edge_to_idx:
                edge_indices.append(self.edge_to_idx[edge])
            elif reverse_edge in self.edge_to_idx:
                edge_indices.append(self.edge_to_idx[reverse_edge])
            else:
                raise ValueError(f"Edge between {from_node} and {to_node} not found")
                
        return edge_indices
        
    def get_path(self, from_idx: int, to_idx: int) -> list[int] | None:
        """
        Get shortest path between two node indices.
        
        Returns:
        --------
        list[int] | None
            List of edge indices or None if no path exists
        """
        return self.paths.get((from_idx, to_idx))
        
    def to_dense_tensor(self, max_path_length: int) -> torch.Tensor:
        """
        Convert to dense tensor representation.
        
        Parameters:
        -----------
        max_path_length : int
            Maximum path length to include. Longer paths are truncated.
            
        Returns:
        --------
        torch.Tensor
            Dense tensor of shape (num_nodes, num_nodes, max_path_length)
            Values are edge indices, -1 for padding/disconnected nodes
        """
        num_nodes = len(self.node_names)
        dense_matrix = torch.full((num_nodes, num_nodes, max_path_length), -1, dtype=torch.int32)
        
        # Fill paths from sparse storage
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    # Diagonal: no edges needed for same node
                    continue
                    
                path = self.get_path(i, j)
                if path is not None:
                    path_length = min(len(path), max_path_length)
                    if path_length > 0:
                        dense_matrix[i, j, :path_length] = torch.tensor(path[:path_length], dtype=torch.int32)
                
        return dense_matrix
        
    def get_max_path_length(self) -> int:
        """
        Get the maximum path length stored.
        
        Returns:
        --------
        int
            Maximum number of edges in any stored path
        """
        if not self.paths:
            return 0
        return max(len(path) for path in self.paths.values() if path is not None)
        
    def get_stats(self) -> dict:
        """
        Get statistics about stored paths.
        
        Returns:
        --------
        dict
            Statistics including number of paths, max/min/avg length
        """
        valid_paths = [path for path in self.paths.values() if path is not None and len(path) > 0]
        
        if not valid_paths:
            return {"num_paths": 0, "max_length": 0, "min_length": 0, "avg_length": 0.0}
            
        lengths = [len(path) for path in valid_paths]
        return {
            "num_paths": len(valid_paths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "disconnected_pairs": len([p for p in self.paths.values() if p is None])
        }

class Dataset(BaseModel):
    """
    Dataset class to hold the dataset information.
    """
    simulator: LeakSimulator
    nodes: dict[str, Node] = {}
    pipes: dict[str, Pipe] = {}
    max_path_length: int = 20
    max_nodes_for_edge_paths: int = 50

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Initialize nodes and pipes after model creation."""
        # __context no se utiliza en esta implementación
        if not self.simulator.simulation_run:
            raise ValueError("Simulation has not been run yet. Please run the simulation before creating the dataset.")
        # Get nodes and pipes from the simulator
        self._get_nodes()
        self._get_pipes()

    def _get_nodes(self) -> None:
        """
        Get the nodes in the water network.
        """
        if self.simulator.results is None:
            raise ValueError("Simulation has not been run yet. Please run the simulation before getting nodes.")

        self.nodes = {}
        for node_name, wn_node in self.simulator.simulator_without_leaks.nodes:
            # Get elevation, using 0 as default for nodes without elevation (like reservoirs)
            # theorical_pressure = self.simulator.results.node['pressure'][node_name]
            # observed_pressure = self.simulator.leak_results.node_pressure(node_name)
            n_type = self._return_wntr_element_type(wn_node)
            head_mean = self.simulator.results['without_leaks'].node['head'][node_name].mean()
            head_std = self.simulator.results['without_leaks'].node['head'][node_name].std()
            theoretical_pressure_mean = self.simulator.results['without_leaks'].node['pressure'][node_name].mean()
            theoretical_pressure_std = self.simulator.results['without_leaks'].node['pressure'][node_name].std()
            observed_pressure_mean = self.simulator.results['with_leaks'].node['pressure'][node_name].mean()
            observed_pressure_std = self.simulator.results['with_leaks'].node['pressure'][node_name].std()

            elevation = getattr(wn_node, 'elevation', 0.0)
            self.nodes[node_name] = Node(
                name=node_name,
                type=n_type,  # 1: junction, 2: tank, 3: reservoir
                elevation=elevation,
                head_mean=head_mean,
                head_std=head_std,
                theoretical_pressure_mean=theoretical_pressure_mean,
                theoretical_pressure_std=theoretical_pressure_std,
                observed_pressure_mean=observed_pressure_mean,
                observed_pressure_std=observed_pressure_std
            )

    def _get_pipes(self) -> None:
        """
        Get the pipes in the water network.
        """
        self.pipes = self.simulator.simulator_without_leaks.pipes

    @property
    def results(self) -> None:
        """
        Get the results of the simulation.
        """
        self.simulator.results

    @property
    def node_name_list(self) -> list[str]:
        """
        List of node names in the water network.
        """
        return self.simulator.simulator_without_leaks.node_name_list

    @property
    def x(self) -> pl.DataFrame:
        """
        `x`has three features:
          - type: int
          - elevation: float
          - head_mean: float
          - head_std: float
          - theorical_pressure_mean: float
          - theorical_pressure_std: float
          - observed_pressure_mean: float
          - observed_pressure_std: float
        """
        return pl.DataFrame({
            "elevation": [node.elevation for node in self.nodes.values()],
            "type": [node.type for node in self.nodes.values()],
            "head_mean": [node.head_mean for node in self.nodes.values()],  # From without leaks
            "head_std": [node.head_std for node in self.nodes.values()],  # From without leaks
            "theoretical_pressure_mean": [node.theoretical_pressure_mean for node in self.nodes.values()],
            "theoretical_pressure_std": [node.theoretical_pressure_std for node in self.nodes.values()],
            "observed_pressure_mean": [node.observed_pressure_mean for node in self.nodes.values()],
            "observed_pressure_std": [node.observed_pressure_std for node in self.nodes.values()]
        })

    @staticmethod
    def _return_wntr_element_type(node: Junction | Tank | Reservoir) -> int:
        """
        Return the type of the node as an integer.
        1: junction, 2: tank, 3: reservoir
        """
        if isinstance(node, Junction):
            return 1
        elif isinstance(node, Tank):
            return 2
        elif isinstance(node, Reservoir):
            return 3
        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def _create_edge_index(self) -> tuple[torch.Tensor, list[tuple[str, str]]]:
        """
        Creates the edge_index tensor for PyG and keeps track of node names.
        
        The edge_index tensor represents the connectivity of the network where each column
        represents an edge [from_node, to_node]. For water networks, we create bidirectional
        edges since water can flow in both directions.
        
        Returns:
        --------
        tuple[torch.Tensor, list[tuple[str, str]]]
            - torch.Tensor: Edge index tensor of shape (2, num_edges)
            - list[tuple[str, str]]: List of (from_node, to_node) name pairs in the same order
        """
        # Create node name to index mapping
        node_to_idx = {name: idx for idx, name in enumerate(self.nodes.keys())}
        
        edge_list = []  # Will store [from_idx, to_idx] pairs
        edge_names = []  # Will store (from_name, to_name) pairs
        
        # Iterate through pipes to create edges
        for pipe in self.pipes.values():
            from_idx = node_to_idx[pipe.from_node]
            to_idx = node_to_idx[pipe.to_node]
            
            # Add forward edge
            edge_list.append([from_idx, to_idx])
            edge_names.append((pipe.from_node, pipe.to_node))
            
            # Add backward edge (unless it's a check valve)
            if not pipe.check_valve:
                edge_list.append([to_idx, from_idx])
                edge_names.append((pipe.to_node, pipe.from_node))
        
        # Convert to tensor of shape (2, num_edges)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return edge_index, edge_names

    @cached_property
    def edge_index(self) -> torch.Tensor:
        """Get the edge index tensor for PyG."""
        edge_index, _ = self._create_edge_index()
        return edge_index

    @cached_property
    def edge_names(self) -> list[tuple[str, str]]:
        """Get the list of edge names in the same order as edge_index."""
        _, edge_names = self._create_edge_index()
        return edge_names
    
    @cached_property
    def edge_attr(self) -> torch.Tensor:
        """
        Get the edge attributes tensor for PyG.
        
        Edge attributes include:
        - length
        - diameter
        - roughness
        - material_code (encoded as an integer)
        - status_code (encoded as an integer)
        
        Returns:
        --------
        torch.Tensor
            Edge attributes tensor of shape (num_edges, num_features)
        """
        return self._create_edge_attributes()

    def _create_edge_attributes(self) -> torch.Tensor:
        """
        Creates the edge attributes tensor for PyG.
        
        Edge attributes are organized as:
        [length, diameter, roughness, material_code, status_code]
        
        Returns:
        --------
        torch.Tensor
            Edge attributes tensor of shape (num_edges, num_features)
        """
        edge_names = self.edge_names  # Get cached edge names
        edge_attrs: list[list[float]] = []
        
        for from_node, to_node in edge_names:
            # Find the corresponding pipe
            pipe = None
            for p in self.pipes.values():
                if (p.from_node == from_node and p.to_node == to_node) or \
                   (not p.check_valve and p.from_node == to_node and p.to_node == from_node):
                    pipe = p
                    break
            
            if pipe is None:
                raise ValueError(f"No pipe found between {from_node} and {to_node}")
            
            # Use the new to_int() method for material encoding
            material_code = pipe.material.to_int() if pipe.material else PipeMaterial.UNKNOWN.to_int()
            
            # Create feature vector for this edge
            edge_attr = [
                float(pipe.length),
                float(pipe.diameter),
                float(pipe.roughness),
                float(material_code),
                float(pipe.status.to_int())  # Use to_int() method for status encoding
            ]
            edge_attrs.append(edge_attr)
        
        return torch.tensor(edge_attrs, dtype=torch.float)

    @cached_property
    def networkx_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph representation of the water network.
        
        Returns:
        --------
        nx.Graph
            Unweighted graph for topological analysis
        """
        return self._create_networkx_graph()
    
    def _create_networkx_graph(self) -> nx.Graph:
        """
        Create NetworkX graph from the water network topology.
        
        Returns:
        --------
        nx.Graph
            Unweighted graph with nodes and edges from the water network
        """
        G = nx.Graph()
        
        # Add nodes
        node_names = list(self.nodes.keys())
        G.add_nodes_from(node_names)
        
        # Add edges without weights for topological analysis
        for pipe in self.pipes.values():
            G.add_edge(pipe.from_node, pipe.to_node)
        
        return G

    @cached_property
    def dist_matrix(self) -> torch.Tensor:
        """
        Get the distance matrix containing shortest path distances between all node pairs.
        
        The distance matrix is computed using NetworkX shortest path algorithms on the
        underlying water network topology. Distances are topological (hop count).
        
        Returns:
        --------
        torch.Tensor
            Symmetric distance matrix of shape (num_nodes, num_nodes) where
            dist_matrix[i][j] represents the shortest path distance from node i to node j
        """
        return self._create_distance_matrix()
    
    def _create_distance_matrix(self) -> torch.Tensor:
        """
        Creates the distance matrix using NetworkX shortest path algorithms.
        Computes topological distances (hop count) instead of physical distances.
        Optimized to only compute upper triangle since matrix is symmetric.
        
        Returns:
        --------
        torch.Tensor
            Distance matrix of shape (num_nodes, num_nodes) with integer distances.
            -1 indicates disconnected nodes.
        """
        # Use cached NetworkX graph
        G = self.networkx_graph
        node_names = list(self.nodes.keys())
        node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        
        n_nodes = len(node_names)
        dist_matrix = np.full((n_nodes, n_nodes), -1, dtype=np.int32)
        
        # Fill diagonal with zeros
        np.fill_diagonal(dist_matrix, 0)
        
        # Only compute upper triangle for optimization
        for i, node_i in enumerate(node_names):
            # Use single-source shortest path for efficiency
            try:
                lengths = nx.single_source_shortest_path_length(G, node_i)
                # Process all nodes reachable from node_i
                for node_j, distance in lengths.items():
                    j = node_to_idx[node_j]
                    if j > i:  # Only fill upper triangle
                        dist_matrix[i, j] = distance
                        dist_matrix[j, i] = distance
            except nx.NetworkXError:
                # Node is isolated, distances remain -1
                pass
        
        return torch.tensor(dist_matrix, dtype=torch.int32)

    @cached_property
    def edge_paths(self) -> EdgePaths:
        """
        Get the EdgePaths object containing shortest paths between all node pairs.
        
        Returns:
        --------
        EdgePaths
            Object containing sparse storage of shortest paths as edge sequences
        """
        return self._create_edge_paths()
    
    def _create_edge_paths(self) -> EdgePaths:
        """
        Create EdgePaths object with shortest paths between all nodes.
        
        Returns:
        --------
        EdgePaths
            EdgePaths object with computed shortest paths
        """
        # Use cached NetworkX graph and node names
        G = self.networkx_graph
        node_names = list(self.nodes.keys())
        edge_names = self.edge_names
        
        # Create and return EdgePaths object
        return EdgePaths(G, node_names, edge_names, max_nodes=self.max_nodes_for_edge_paths)
    
    def get_edge_paths_tensor(self, max_path_length: int | None = None) -> torch.Tensor:
        """
        Get edge paths as dense tensor with specified maximum path length.
        
        Parameters:
        -----------
        max_path_length : int | None, default=None
            Maximum path length to include in tensor. If None, uses self.max_path_length
            
        Returns:
        --------
        torch.Tensor
            Dense tensor of shape (num_nodes, num_nodes, max_path_length)
            containing edge indices for shortest paths
        """
        if max_path_length is None:
            max_path_length = self.max_path_length
        return self.edge_paths.to_dense_tensor(max_path_length)


# MISSING

# Sensor Mask: Binary mask indicating which nodes have sensor measurements

"""
We need to create the dataset for the leak simulator.
It has the following attributes:
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        dist_matrix: torch.Tensor,
        edge_paths: torch.Tensor,
        sensor_mask_node: torch.Tensor,
        leak_targets: torch.Tensor,

We'll have to split the network, but the best appraoch is to do it
later. 

We fisrt create the Nodes types and pipes and then from there we
create the dataset.

Afterwards we will split the netowork in different parts using
the algorithms from the networkx library.

This split has to tanslet in different datasets.

The dataset class In the repo of the data has to be able to permutate itself

"""



