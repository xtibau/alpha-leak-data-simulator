from pydantic import BaseModel, ConfigDict
from functools import cached_property
import polars as pl
from wntr.network.elements import Junction, Tank, Reservoir
import torch

from alpha_leak_data.data import AlphaLeakData

from water_netowrk_simulator.enums import PipeMaterial
from water_netowrk_simulator.leak_simulator import LeakSimulator

from ..models import Node, Pipe

class Dataset(BaseModel):
    """
    Dataset class to hold the dataset information.
    """
    simulator: LeakSimulator
    nodes: dict[str, Node] = {}
    pipes: dict[str, Pipe] = {}

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
                float(pipe.status.value)  # PipeStatus is still an IntEnum
            ]
            edge_attrs.append(edge_attr)
        
        return torch.tensor(edge_attrs, dtype=torch.float)


MISSING 
Distance Matrix: Contains shortest path distances between nodes

Edge Paths: Contains the sequence of edge IDs along the shortest path between nodes

Sensor Mask: Binary mask indicating which nodes have sensor measurements

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



