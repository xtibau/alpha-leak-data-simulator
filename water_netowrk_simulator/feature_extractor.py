from water_netowrk_simulator.dataset.dataset import Dataset


class FeatureExtractor:
    def __init__(self, dataset: Dataset):
        """
        Initialize the feature extractor.
        The feature extractor must create the features of the network. 
        
        Parameters:
        -----------
        water_network : wntr.network.WaterNetworkModel
            Water network model
        """
        self.dataset = dataset
        
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
