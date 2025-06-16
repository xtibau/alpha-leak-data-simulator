from enum import Enum

import numpy as np

class SimulatorType(Enum):
    WNTRS = "WNTRS"
    EPANET = "EPANET"

    def __str__(self):
        return self.value

class PipeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value
    
    def to_int(self) -> int:
        """
        Convert the pipe status to an integer representation.
        
        Returns:
        --------
        int
            Integer code for the pipe status: 0=closed, 1=open, -1=unknown
        """
        status_to_int: dict[PipeStatus, int] = {
            PipeStatus.CLOSED: 0,
            PipeStatus.OPEN: 1,
            PipeStatus.UNKNOWN: -1
        }
        return status_to_int[self]

# On the futer to consider changes of leak from material type.
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
    UNKNOWN = "UNKNOWN"

    def __str__(self):
        return self.value
    
    def to_int(self) -> int:
        """
        Convert the material type to an integer representation.
        
        Returns:
        --------
        int
            Integer code for the material type, starting from 0 for UNKNOWN
        """
        # Define mapping of materials to integers
        material_to_int: dict[PipeMaterial, int] = {
            PipeMaterial.UNKNOWN: 0,
            PipeMaterial.PVC: 1,
            PipeMaterial.POLYETHYLENE: 2,
            PipeMaterial.IRON: 3,
            PipeMaterial.CAST_IRON: 4,
            PipeMaterial.DUCTILE_IRON: 5,
            PipeMaterial.STEEL: 6,
            PipeMaterial.CONCRETE: 7,
            PipeMaterial.ASBESTOS_CEMENT: 8,
            PipeMaterial.OTHER: 9
        }
        return material_to_int[self]

class LeakSeverity(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    BURST = "burst"

    @classmethod
    def get_random_severity(cls, severity_distribution):
        """
        Get a random severity based on the given distribution.
        
        Parameters:
        -----------
        severity_distribution : dict[LeakSeverity, float]
            Dictionary mapping LeakSeverity values to their probabilities.
            Probabilities should sum to 1.0.
        
        Returns:
        --------
        LeakSeverity
            A randomly selected severity based on the provided distribution.
        """
        # Extract the severities and their probabilities
        severities = list(severity_distribution.keys())
        probabilities = list(severity_distribution.values())
        
        # Validate probabilities sum to approximately 1
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.0")
            
        # Get a random severity based on the distribution
        return np.random.choice(severities, p=probabilities)

    def __str__(self):
        return self.value
