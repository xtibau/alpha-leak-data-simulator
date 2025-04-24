from enum import Enum
import numpy as np
import wntr
import pandas as pd

from wntr.network import WaterNetworkModel, Pattern

class SimulatorType(Enum):
    WNTRS = "WNTRS"
    EPANET = "EPANET"


class WaterNetworksimulator:
    """
    A class to simulate nighttime pressure conditions in a water network using WNTR.
    """
    
    def __init__(self, inp_file_path: str,
                 sim_type: SimulatorType = SimulatorType.WNTRS,
                 verbose: bool = False) -> None:
        """
        Initialize the simulator with an INP file.
        
        Parameters:
        -----------
        inp_file_path : str
            Path to the EPANET INP file
        sim_type : SimulatorType, optional
            Type of simulator to use (WNTRS or EPANET)
        verbose : bool, optional
            Whether to print detailed information
        """
        self.inp_file_path: str = inp_file_path
        self.sim_type: SimulatorType = sim_type
        self.verbose: bool = verbose

        # Initialize attributes
        self.wn: wntr.network.WaterNetworkModel | None = None
        self.sim: wntr.sim.WNTRSimulator | wntr.sim.EpanetSimulator | None = None
        self.results: wntr.sim.SimulationResults | None = None
        
        # Load the water network and setup the simulator
        self.load_network()
        self._set_simulator(sim_type)
    
    # Network setup and configuration methods
    def load_network(self) -> None:
        """Load the water network from the INP file."""
        try:
            self.wn: WaterNetworkModel = wntr.network.WaterNetworkModel(self.inp_file_path)
            if self.verbose:
                print(f"Loaded network with {len(self.wn.junctions())} junctions and "
                      f"{len(self.wn.tanks())} tanks.")
        except Exception as e:
            print(f"Error loading network: {str(e)}")
            raise
    
    def _set_simulator(self, sim_type: SimulatorType = SimulatorType.WNTRS) -> None:
        match sim_type:
            case SimulatorType.WNTRS:
                self.sim = wntr.sim.WNTRSimulator(self.wn)
            case SimulatorType.EPANET:
                self.sim = wntr.sim.EpanetSimulator(self.wn)
    
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
        for tank_name, tank in self.wn.tanks().items():
            if level is not None:
                tank.init_level = level
            else:
                # Calculate level based on percentage of tank capacity
                tank_capacity = tank.max_level - tank.min_level
                tank.init_level = tank.min_level + (tank_capacity * fill_percent / 100)
            
        print(f"Tank levels set for simulation")
    
    @property
    def nodes(self):
        """
        Get all nodes in the network.
        
        Returns:
        --------
        list
            List of nodes in the network
        """
        return self.wn.nodes()
    
    @property
    def node_name_list(self):
        """
        Get all node IDs in the network.
        
        Returns:
        --------
        list
            List of node IDs in the network
        """
        return self.wn.node_name_list

    # Demand pattern methods
    def add_random_demand_noise(self,
                                demand_name : str = "base_demand",
                                prob_noise: float = 0.1,
                                min_demand: float = 0.001, 
                                max_demand: float = 0.005) -> None:
        """
        Assigns a random demand to each junction in the network.
        This simulates a low base demand, representing nighttime consumption.
        The demand is randomly generated within the specified range.

        Parameters:
        -----------
        demand_name : str, optional
            Name for the demand pattern. Defaults to "base_demand"
        prob_noise : float, optional
            Probability of adding noise to a junction. Defaults to 0.1
        min_demand : float, optional
            Minimum demand value (m³/s). Defaults to 0.001
        max_demand : float, optional
            Maximum demand value (m³/s). Defaults to 0.005
        """
        for junction_name, junction in self.wn.junctions():
            if np.random.rand() < prob_noise:
                base_demand = np.random.uniform(min_demand, max_demand)
            else:
                base_demand = 0.0
            # Clear existing demand timeseries
            junction.demand_timeseries_list.clear()
            # Add new demand timeseries
            junction.add_demand(base=base_demand,
                                pattern_name=demand_name,
                                category='noise_demand')

            if self.verbose:
                print(f"Junta {junction_name}: demanda base asignada de {base_demand:.4f} m³/s")
    
    def add_nighttime_pattern(self, pattern_name: str = "night_pattern",
                              base_demand: int = 1,
                              pattern_hours: int = 6,
                              start_hour: int = 0) -> None:
        """
        Creates a nighttime demand pattern with constant multipliers.
        
        Parameters:
        -----------
        pattern_name : str, optional
            Name of the pattern to create. Defaults to "night_pattern"
        base_demand : int, optional
            Base demand multiplier. Defaults to 1
        pattern_hours : int, optional
            Duration of the pattern in hours. Defaults to 6
        start_hour : int, optional
            Starting hour of the pattern. Defaults to 0 (midnight)
        
        Raises:
        -------
        ValueError
            If junctions don't have base demand assigned
        """
        if self.wn is None:
            print("Error: Water network model not loaded.")
            return

        pattern_multipliers: list = [base_demand] * pattern_hours
        self.wn.add_pattern(pattern_name, pattern_multipliers)
        pattern = self.wn.get_pattern(pattern_name)
        pattern.pattern_step = pd.Timedelta(hours=1)
        pattern.start_time = pd.Timedelta(hours=start_hour)
        
        for junction_name, junction in self.wn.junctions():
            if junction.demand_timeseries_list:
                junction.demand_timeseries_list[0].pattern_name = pattern_name
            else:
                # There is no base demand
                raise ValueError(f"Junction {junction_name} has no base demand assigned.")
        
        if self.verbose:
            print(f"Junctions assigned to pattern '{pattern_name}'")
    
    # Simulation and results methods
    def run_simulation(self) -> None:
        """
        Run a hydraulic simulation.
        
        Returns:
        --------
        None
            Results are stored in self.results attribute
        """
        if self.sim is None:
            self._set_simulator()
        
        self.results = self.sim.run_sim()
        
        print("Simulation completed")

