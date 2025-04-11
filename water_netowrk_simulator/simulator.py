from enum import Enum
import wntr
import pandas as pd

from wntr.network import WaterNetworkModel

class SimulatorType(Enum):
    WNTRS = "WNTRS"
    EPANET = "EPANET"


class WaterNetworksimulator:
    """
    A class to simulate nighttime pressure conditions in a water network using WNTR.
    """
    
    def __init__(self, inp_file_path: str,
                 sim_type: SimulatorType = SimulatorType.WNTRS) -> None:
        """
        Initialize the simulator with an INP file.
        
        Parameters:
        -----------
        inp_file_path : str
            Path to the EPANET INP file
        sim_type : SimulatorType, optional
            Type of simulator to use (WNTRS or EPANET)
        """
        self.inp_file_path: str = inp_file_path
        self.sim_type: SimulatorType = sim_type

        # Initialize attributes
        self.wn: wntr.network.WaterNetworkModel | None = None
        self.sim: wntr.sim.WNTRSimulator | wntr.sim.EpanetSimulator | None = None
        self.results: wntr.sim.SimulationResults | None = None
        
        # Load the water network
        self._set_simulator(sim_type)
        self.load_network()

    
    def load_network(self) -> None:
        """Load the water network from the INP file."""
        try:
            self.wn: WaterNetworkModel = wntr.network.WaterNetworkModel(self.inp_file_path)
            print(f"Successfully loaded network from {self.inp_file_path}")
        except Exception as e:
            print(f"Error loading network: {str(e)}")
            raise
    
    def set_zero_demands(self) -> None:
        """Set all junction demands to zero to simulate nighttime conditions."""
        for _, junction in self.wn.junctions():
            junction.demand_timeseries_list[0].base_value = 0.0
        print("All junction demands set to zero")
    
    def set_tank_levels(self, level: float | None = None, fill_percent: float | None = 75) -> None:
        """
        Set tank levels for the simulation.
        
        Parameters:
        -----------
        level : float, optional
            Specific water level to set for all tanks
        fill_percent : float, optional
            Percentage of tank capacity to fill (0-100%)
        """
            
        for tank_name, tank in self.wn.tanks():
            if level is not None:
                tank.init_level = level
            else:
                # Calculate level based on percentage of tank capacity
                tank_capacity = tank.max_level - tank.min_level
                tank.init_level = tank.min_level + (tank_capacity * fill_percent / 100)
            
        print(f"Tank levels set for simulation")

    def _set_simulator(self, sim_type: SimulatorType = SimulatorType.WNTRS) -> None:

        match sim_type:
            case SimulatorType.WNTRS:
                self.sim = wntr.sim.WNTRSimulator(self.wn)
            case SimulatorType.EPANET:
                self.sim = wntr.sim.EpanetSimulator(self.wn)

    def run_simulation(self) -> None:
        """
        Run a shydraulic simulation.
        
        Returns:
        --------
        pressure_results : pandas.DataFrame
            DataFrame containing pressures at all junctions
        """
        if self.sim is None:
            self._set_simulator()
        
        self.results = self.sim.run_sim()
        
        print("Simulation completed")
    
    def get_link_pressures(self, node_pressures: pd.Series) -> dict:
        """
        Calculate the approximate pressure at each link by averaging the pressures
        at the start and end nodes of the link.
        
        Parameters:
        -----------
        node_pressures : pandas.Series
            Pressures at nodes/junctions
            
        Returns:
        --------
        link_pressures : dict
            Dictionary with link names as keys and estimated pressures as values
        """
        link_pressures = {}
        
        for link_name, link in self.wn.links():
            start_node = link.start_node_name
            end_node = link.end_node_name
            
            # Check if both nodes have pressure values
            if start_node in node_pressures.index and end_node in node_pressures.index:
                start_pressure = node_pressures[start_node]
                end_pressure = node_pressures[end_node]
                # Estimate link pressure as average of endpoint pressures
                link_pressures[link_name] = (start_pressure + end_pressure) / 2
            
        return link_pressures

    def add_leak(self, node_id: str, leak_area: float, discharge_coeff: float = 0.75, leak_type: str = 'leak') -> None:
        """
        Add a leak to the specified node.
        
        Parameters:
        -----------
        node_id : str
            ID of the node where leak will be added
        leak_area : float
            Area of the leak in m^2
        discharge_coeff : float, optional
            Discharge coefficient
        leak_type : str, optional
            Type of leak ('leak' or 'demand')
        """
        # Implementation based on WNTR documentation
        # https://github.com/usepa/WNTR/blob/main/documentation/hydraulics.rst

        raise NotImplementedError
        
    def add_random_demand_noise(self, mean: float = 0.0, std_dev: float = 0.01, percentage_of_nodes: float = 0.2) -> None:
        """
        Add small random demand to a percentage of nodes.
        
        Parameters:
        -----------
        mean : float
            Mean of the normal distribution for demand noise
        std_dev : float
            Standard deviation of the normal distribution for demand noise
        percentage_of_nodes : float
            Percentage of nodes to which noise will be added (0-1)
        """
        # Implementation to add small random demands        
        raise NotImplementedError
        
    
    def save_results(self, pressures: pd.Series | pd.DataFrame | dict, output_file: str) -> None:
        """
        Save pressure results to a CSV file.
        
        Parameters:
        -----------
        pressures : pandas.Series, pandas.DataFrame, or dict
            Pressures at nodes or links
        output_file : str
            Path to save the results
        """
        if isinstance(pressures, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame.from_dict(pressures, orient='index', columns=['Pressure'])
        else:
            # Convert Series to DataFrame if needed
            df = pressures.to_frame(name='Pressure') if isinstance(pressures, pd.Series) else pressures
        
        df.index.name = 'Element'
        df.to_csv(output_file)
        print(f"Results saved to {output_file}")

