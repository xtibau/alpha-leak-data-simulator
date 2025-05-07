from enum import Enum

class SimulatorType(Enum):
    WNTRS = "WNTRS"
    EPANET = "EPANET"

class PipeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"

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