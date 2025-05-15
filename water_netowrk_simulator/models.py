from pydantic import BaseModel
from functools import cached_property
import numpy as np

from .enums import LeakSeverity, PipeStatus, PipeMaterial, SimulatorType

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
    severity: LeakSeverity
    area_percent: float
    area: float

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
    
class DemandNoise(BaseModel):
    """
    A class to represent demand noise in the water network.
    """
    name: str = "demand_noise"
    prob_noise: float = 0.1
    min_demand: float = 0.001
    max_demand: float = 0.005

class Config(BaseModel):
    """
    Configuration for the Simulator class.
    
    Attributes:
    -----------
    arbitrary_types_allowed : bool
        Allow arbitrary types in the model
    """
    sim_type: SimulatorType = SimulatorType.WNTRS
    tank_fill_percent: float = 75
    demand_name: str = "demand"
    demand_noise: None | dict[str, float] = None
    night_pattern_hours: int = 6
    severity_distribution: dict[LeakSeverity, float] = {
            LeakSeverity.SMALL: 0.4,
            LeakSeverity.MEDIUM: 0.3,
            LeakSeverity.LARGE: 0.2,
            LeakSeverity.BURST: 0.1,
    }
    leak_probability: float = 0.005
    leak_area_percent: dict[LeakSeverity, tuple[float, float]] = {
            LeakSeverity.SMALL: (0.01, 0.05),
            LeakSeverity.MEDIUM: (0.05, 0.15),
            LeakSeverity.LARGE: (0.15, 0.15),
            LeakSeverity.BURST: (0.20, 0.45)
    }
