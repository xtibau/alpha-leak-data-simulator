import numpy as np
from wntr.sim.results import SimulationResults

from .simulator import WaterNetworksimulator
from .enums import LeakSeverity
from .models import Config, Leak, Pipe

class LeakSimulator:
    """
    A class to simulate leaks in water networks, building on the base WaterNetworksimulator.
    Takes two instances of the simulator: one for the baseline simulation without leaks,
    and another for the simulation with leaks.

    """
    
    def __init__(self, 
                 inp_file_path: str, 
                 config: Config,
                 seed: None | int = None,
                 verbose: bool = False) -> None:
        """
        Initialize the leak simulator.
        
        Parameters:
        -----------
        inp_file_path : str
            Path to the EPANET INP file
        sim_type : SimulatorType, optional
            Type of simulator to use (WNTRS or EPANET)
        seed : int, optional
            Random seed for reproducibility
        verbose : bool, optional
            Whether to print detailed information
        """
        
        # Initialize attributes
        self.inp_file_path = inp_file_path
        self.config = config
        self.sim_type = config.sim_type
        self.verbose = verbose
        self.seed = seed

        # Initialize base simulator without leaks
        self.simulator_without_leaks = WaterNetworksimulator(inp_file_path, self.sim_type, verbose)
        self.simulator_with_leaks = WaterNetworksimulator(inp_file_path, self.sim_type, verbose)

        # We use the reference the same with leaks,.
        self.pies = self.simulator_with_leaks.pipes
        self.pipes_list = self.simulator_with_leaks.pipes_list

        self.results: dict[str, SimulationResults] = {}
        self.simulation_run: bool = False
        
        if seed is not None:
            np.random.seed(seed)

    def establish_leaks(self) -> None:
        """
        Decides the pipes that will be leaking.
        Iterates over the pipes and randomly assigns a leak.
        """

        for pipe in self.pipes_list:
            # Randomly decide if the pipe should leak
            if np.random.rand() < self.config.leak_probability:
                # Add a leak to the pipe
                severity: LeakSeverity = LeakSeverity.get_random_severity(self.config.severity_distribution)
                area_percent, leak_area = self._calculate_leak_area(pipe, severity)
                pipe.leak = Leak(name="Leak_" + pipe.name,
                                 severity=severity,
                                 area_percent=area_percent,
                                 area=leak_area,
                                 )

                if self.verbose:
                    print(f"Added leak to pipe {pipe.name}")

    def _calculate_leak_area(self, pipe: Pipe, severity: LeakSeverity) -> tuple[float, float]:
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
        area_percent = np.random.uniform(min_percent, max_percent)
        
        # Calculate pipe cross-sectional area
        pipe_diameter = pipe.diameter
        pipe_area = np.pi * (pipe_diameter/2)**2
        
        # Calculate leak area
        return area_percent, pipe_area * area_percent
     
    def add_leaks(self) -> None:
        """
        Adds leaks to the simulator with leaks.
        """
        for pipe in self.pipes_list:
            if pipe.leak is not None:
                # Add leak to the simulator with leaks
                self.simulator_with_leaks.add_leak(pipe=pipe)
                
                if self.verbose:
                    print(f"Added leak to simulator with leaks for pipe {pipe.name}")

    def run_simulation(self) -> None:
        """
        Runs a simulation using a simulator.
        """

        for sim in [self.simulator_without_leaks, self.simulator_with_leaks]:
            # Set tank levels
            sim.set_tank_levels(fill_percent=self.config.tank_fill_percent)
            
            # Add random demand noise if requested
            if self.config.demand_noise is not None:
                sim.add_random_demand_noise(
                    demand_name=self.config.demand_name,
                    prob_noise=self.config.demand_noise['prob_noise'],
                    min_demand=self.config.demand_noise['min_demand'],
                    max_demand=self.config.demand_noise['max_demand']
                )                
                # Add nighttime pattern
                sim.add_nighttime_pattern(pattern_hours=self.config.night_pattern_hours)

            if self.verbose:
                print("Baseline simulation setup complete")
            
            # Set simulation duration
            sim.wn.options.time.duration = self.config.night_pattern_hours * 3600

            print("Running simulation...")

            sim.run_simulation()

        self.results_without_leaks = self.simulator_without_leaks.results
        self.results_with_leaks = self.simulator_with_leaks.results

        self.results['without_leaks'] = self.results_without_leaks
        self.results['with_leaks'] = self.results_with_leaks
        self.simulation_run = True
