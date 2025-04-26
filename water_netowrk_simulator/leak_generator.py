import wntr
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any


# Enums for pipe materials and leak severity
class PipeMaterial(Enum):
    PVC = "PVC"
    POLYETHYLENE = "POLYETHYLENE"
    IRON = "IRON"
    CAST_IRON = "CAST_IRON"
    DUCTILE_IRON = "DUCTILE_IRON"
    STEEL = "STEEL"
    CONCRETE = "CONCRETE"
    ASBESTOS_CEMENT = "ASBESTOS_CEMENT"
    OTHER = "OTHER"


class LeakSeverity(Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    BURST = "BURST"


class LeakConfig:
    """Configuration for leak properties."""
    
    def __init__(
        self, 
        discharge_coefficients: Dict[PipeMaterial, Tuple[float, float]], 
        leak_area_percent: Dict[LeakSeverity, Tuple[float, float]], 
        leak_probability: float = 0.05
    ) -> None:
        """
        Initialize leak configuration.
        
        Args:
            discharge_coefficients: Discharge coefficients for different pipe materials
                                   Format: {PipeMaterial: (min_coeff, max_coeff)}
            leak_area_percent: Leak area as percentage of pipe cross-section for different severities
                              Format: {LeakSeverity: (min_percent, max_percent)}
            leak_probability: Base probability for a pipe to have a leak (0-1)
        """
        self.discharge_coefficients = discharge_coefficients
        self.leak_area_percent = leak_area_percent
        self.leak_probability = leak_probability


class LeakGenerator:
    """
    Class for generating and creating leaks in a water network model.
    Uses a separate LeakConfig object for configuration.
    """
    
    def __init__(
        self, 
        network: 'wntr.network.WaterNetworkModel', 
        config: LeakConfig,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the leak generator.
        
        Args:
            network: The water network model
            config: Configuration for leak properties
            seed: Random seed for reproducibility
        """
        self.network = network
        self.config = config
        self.rng = np.random.RandomState(seed)
    
    def should_pipe_have_leak(self, pipe_name: str) -> bool:
        """
        Determine if a given pipe should have a leak based on probability.
        
        Args:
            pipe_name: Name of the pipe to check
            
        Returns:
            True if the pipe should have a leak, False otherwise
        """
        # Basic probability check - can be extended with more sophisticated logic
        return self.rng.random() < self.config.leak_probability
    
    def get_pipes_with_leaks(self) -> List[str]:
        """
        Return a list of all pipes that should have leaks based on probability.
        
        Returns:
            Names of pipes that should have leaks
        """
        leaky_pipes: List[str] = []
        for pipe_name in self.network.pipe_name_list:
            if self.should_pipe_have_leak(pipe_name):
                leaky_pipes.append(pipe_name)
        return leaky_pipes
    
    def get_specific_number_of_leaky_pipes(self, num_leaks: int) -> List[str]:
        """
        Get a specific number of pipes that should have leaks.
        
        Args:
            num_leaks: Number of leaks to generate
            
        Returns:
            Names of pipes that should have leaks
        """
        # Ensure we don't try to create more leaks than there are pipes
        num_leaks = min(num_leaks, len(self.network.pipe_name_list))
        
        # Randomly select pipes
        selected_pipes = self.rng.choice(
            self.network.pipe_name_list, 
            size=num_leaks, 
            replace=False
        )
        
        return list(selected_pipes)
    
    def _get_pipe_material(self, pipe: Any) -> PipeMaterial:
        """
        Determine pipe material based on pipe properties.
        
        Args:
            pipe: Pipe object from the network
            
        Returns:
            Pipe material enum
        """
        # This is a simplified approach
        # In a real implementation, you would extract material from pipe properties
        # or use a mapping based on pipe characteristics
        return PipeMaterial.OTHER  # Default fallback
    
    def _get_random_severity(self) -> LeakSeverity:
        """
        Choose a random leak severity.
        
        Returns:
            Random leak severity
        """
        severities = list(LeakSeverity)
        return self.rng.choice(severities)
    
    def _calculate_leak_area(self, pipe: Any, severity: LeakSeverity) -> float:
        """
        Calculate leak area based on pipe dimensions and severity.
        
        Args:
            pipe: Pipe object from the network
            severity: Severity of the leak
            
        Returns:
            Leak area in m²
        """
        # Get the range for the specified severity
        min_percent, max_percent = self.config.leak_area_percent[severity]
        
        # Choose a random percentage within the range
        area_percent = self.rng.uniform(min_percent, max_percent)
        
        # Calculate pipe cross-sectional area
        pipe_diameter = pipe.diameter
        pipe_area = np.pi * (pipe_diameter/2)**2
        
        # Calculate leak area
        leak_area = pipe_area * area_percent
        return leak_area
    
    def _get_discharge_coefficient(self, material: PipeMaterial) -> float:
        """
        Get discharge coefficient based on pipe material.
        
        Args:
            material: Pipe material
            
        Returns:
            Discharge coefficient
        """
        min_coeff, max_coeff = self.config.discharge_coefficients[material]
        return self.rng.uniform(min_coeff, max_coeff)
    
    def _add_leak_to_network(
        self, 
        pipe_name: str, 
        leak_area: float, 
        discharge_coeff: float, 
        start_time: int = 0, 
        end_time: Optional[int] = None
    ) -> str:
        """
        Add a leak to the network by creating a new junction and adjusting the pipe.
        
        Uses the official WNTR morph functions to split the pipe and add a leak.
        
        Args:
            pipe_name: Name of the pipe to add a leak to
            leak_area: Area of the leak in m²
            discharge_coeff: Discharge coefficient
            start_time: Start time of the leak in seconds. Default is 0.
            end_time: End time of the leak in seconds. If None, leak is permanent.
            
        Returns:
            Name of the leak junction
        """
        # Generate unique names for the new pipe and junction
        new_pipe_name = f"{pipe_name}_B"
        junction_name = f"leak_{pipe_name}"
        
        # Split the pipe using WNTR's built-in function
        # This creates a new junction and splits the pipe into two segments
        self.network = wntr.morph.split_pipe(
            self.network, 
            pipe_name, 
            new_pipe_name, 
            junction_name
        )
        
        # Get the newly created junction
        leak_node = self.network.get_node(junction_name)
        
        # Add the leak to the junction using WNTR's add_leak method
        # This properly handles the leak hydraulics including the discharge coefficient
        leak_node.add_leak(
            self.network, 
            area=leak_area,
            discharge_coeff=discharge_coeff,
            start_time=start_time,
            end_time=end_time
        )
        
        return junction_name
    
    def create_leak(
        self, 
        pipe_name: str, 
        severity: Optional[LeakSeverity] = None
    ) -> Dict[str, Any]:
        """
        Create a leak on a specific pipe.
        
        Args:
            pipe_name: Name of the pipe to add a leak to
            severity: Severity of the leak. If None, a random severity is chosen.
            
        Returns:
            Information about the created leak
        """
        # Get the pipe object
        pipe = self.network.get_link(pipe_name)
        
        # Determine pipe material
        pipe_material = self._get_pipe_material(pipe)
        
        # Determine leak severity if not specified
        if severity is None:
            severity = self._get_random_severity()
        
        # Calculate leak properties
        leak_area = self._calculate_leak_area(pipe, severity)
        discharge_coeff = self._get_discharge_coefficient(pipe_material)
        
        # Add the leak to the network
        junction_name = self._add_leak_to_network(pipe_name, leak_area, discharge_coeff)
        
        # Return information about the leak
        return {
            'pipe_name': pipe_name,
            'severity': severity.value,
            'leak_area': leak_area,
            'discharge_coeff': discharge_coeff,
            'junction_name': junction_name
        }
    
    def create_leaks_for_pipes(self, pipe_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Create leaks for a list of pipes.
        
        Args:
            pipe_names: List of pipe names to add leaks to
            
        Returns:
            Mapping of leak junction names to leak information
        """
        leak_info: Dict[str, Dict[str, Any]] = {}
        for pipe_name in pipe_names:
            leak_data = self.create_leak(pipe_name)
            leak_info[leak_data['junction_name']] = leak_data
        return leak_info
    
    def generate_random_leaks(self, num_leaks: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate random leaks in the network.
        
        Args:
            num_leaks: Number of leaks to generate. If None, uses probability-based approach.
            
        Returns:
            Mapping of leak junction names to leak information
        """
        if num_leaks is not None:
            # Generate a specific number of leaks
            pipes_to_leak = self.get_specific_number_of_leaky_pipes(num_leaks)
        else:
            # Generate leaks based on probability
            pipes_to_leak = self.get_pipes_with_leaks()
        
        return self.create_leaks_for_pipes(pipes_to_leak)


# Example usage
def example_usage() -> None:
    """
    Example of how to use the LeakConfig and LeakGenerator classes.
    """
    # Import network model
    wn = wntr.network.WaterNetworkModel('example.inp')
    
    # Create leak configuration
    leak_config = LeakConfig(
        discharge_coefficients={
            PipeMaterial.PVC: (0.6, 0.8),
            PipeMaterial.IRON: (0.5, 0.7),
            PipeMaterial.OTHER: (0.4, 0.6)
            # Add more materials as needed
        },
        leak_area_percent={
            LeakSeverity.SMALL: (0.001, 0.01),
            LeakSeverity.MEDIUM: (0.01, 0.05),
            LeakSeverity.LARGE: (0.05, 0.15),
            LeakSeverity.BURST: (0.15, 0.5)
        },
        leak_probability=0.05
    )
    
    # Create leak generator
    generator = LeakGenerator(wn, leak_config, seed=42)
    
    # Generate random leaks based on probability
    random_leaks = generator.generate_random_leaks()
    print(f"Generated {len(random_leaks)} random leaks")
    
    # Generate a specific number of leaks
    specific_leaks = generator.generate_random_leaks(num_leaks=5)
    print(f"Generated {len(specific_leaks)} specific leaks")
    
    # Create a leak on a specific pipe with a specific severity
    pipe_name = wn.pipe_name_list[0]
    leak_info = generator.create_leak(pipe_name, severity=LeakSeverity.LARGE)
    print(f"Created leak on pipe {pipe_name} with severity {leak_info['severity']}")
