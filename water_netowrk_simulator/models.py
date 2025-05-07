from pydantic import BaseModel
from functools import cached_property

from .enums import PipeStatus, PipeMaterial

class Leak(BaseModel):
    """
    A class to represent a leak in the water network.
    
    Attributes:
    -----------
    name : str
        Name of the leak
    location : str
        Location of the leak
    diameter : float
        Diameter of the leak (m)
    flow_rate : float
        Flow rate of the leak (m³/s)
    """
    name: str
    area_percent: float
    leak_area: float

class Pipe(BaseModel):
    """
    A class to represent a pipe in the water network.
    
    Attributes:
    -----------
    name : str
        Name of the pipe
    length : float
        Length of the pipe (m)
    diameter : float
        Diameter of the pipe (m)
    roughness : float
        Roughness coefficient of the pipe
    """
    name: str
    from_node: str
    to_node: str
    minor_loss: float
    check_valve: bool
    status: PipeStatus
    length: float
    diameter: float
    roughness: float
    material: PipeMaterial = PipeMaterial.UNKNOWN
    leak: None | Leak = None

    @cached_property
    def area(self) -> float:
        """
        Calculate the cross-sectional area of the pipe.
        
        Returns:
        --------
        float
            Cross-sectional area of the pipe (m²)
        """
        return np.pi * (self.diameter / 2) ** 2