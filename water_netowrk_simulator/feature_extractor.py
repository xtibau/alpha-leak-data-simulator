class FeatureExtractor:
    def __init__(self, water_network):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        water_network : wntr.network.WaterNetworkModel
            Water network model
        """
        self.wn = water_network
        
    def extract_node_features(self, simulation_results=None):
        """
        Extract features for all nodes.
        
        Parameters:
        -----------
        simulation_results : wntr.sim.results.SimulationResults, optional
            Results from a simulation
            
        Returns:
        --------
        node_features : pandas.DataFrame
            DataFrame with node features
        """
        # Extract features such as:
        # - Altitude (elevation)
        # - Head
        # - Type (Reservoir, Tank, Junction)
        # - Minor-loss
        # - Theoretical pressure
        # - True pressure
        # - Flow (cabal)
        # - Other INP file features

        raise NotImplementedError
        
    def extract_link_features(self, simulation_results=None):
        """
        Extract features for all links.
        
        Parameters:
        -----------
        simulation_results : wntr.sim.results.SimulationResults, optional
            Results from a simulation
            
        Returns:
        --------
        link_features : pandas.DataFrame
            DataFrame with link features
        """
        # Extract features such as:
        # - Material
        # - Diameter
        # - Length
        # - Roughness coefficient
        # - Other INP file features
        raise NotImplementedError

    def create_adjacency_matrix(self):
        """
        Create adjacency matrix for the network.
        
        Returns:
        --------
        adj_matrix : numpy.ndarray
            NxN adjacency matrix
        """
        # Implementation for adjacency matrix creation
        raise NotImplementedError

    def create_distance_matrix_nodes(self):
        """
        Create distance matrix between nodes.
        
        Returns:
        --------
        dist_matrix : numpy.ndarray
            NxN matrix with minimum distance between nodes
        """
        # Use networkx or custom implementation to compute
        # shortest paths between all pairs of nodes
        raise NotImplementedError

    def create_distance_matrix_edges(self):
        """
        Create distance matrix between edges.
        
        Returns:
        --------
        dist_matrix : numpy.ndarray
            ExE matrix with minimum distance between edges
        """
        # Implementation for edge distance matrix
        raise NotImplementedError
