class DataAugmentor:
    def __init__(self):
        """Initialize the data augmentor."""
        pass
        
    def permute_node_order(self, node_features, adjacency_matrix, 
                           node_distance_matrix):
        """
        Permute the order of nodes in features and matrices.
        
        Parameters:
        -----------
        node_features : pandas.DataFrame
            DataFrame with node features
        adjacency_matrix : numpy.ndarray
            NxN adjacency matrix
        node_distance_matrix : numpy.ndarray
            NxN distance matrix
            
        Returns:
        --------
        permuted_data : dict
            Dictionary with permuted data
        """
        # Implementation for node permutation
        raise NotImplementedError
    
    def permute_edge_order(self, edge_features, edge_distance_matrix):
        """
        Permute the order of edges in features and matrices.
        
        Parameters:
        -----------
        edge_features : pandas.DataFrame
            DataFrame with edge features
        edge_distance_matrix : numpy.ndarray
            ExE distance matrix
            
        Returns:
        --------
        permuted_data : dict
            Dictionary with permuted data
        """
        # Implementation for edge permutation
        raise NotImplementedError