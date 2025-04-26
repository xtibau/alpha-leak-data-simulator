import numpy as np
import wntr
from typing import Dict, List, Optional, Tuple, Union

from .simulator import WaterNetworksimulator, SimulatorType
from .leak_generator import LeakGenerator, LeakSeverity

class LeakSimulator:
    """
    A class to simulate leaks in water networks, building on the base WaterNetworksimulator.
    """
    
    def __init__(self, 
                 inp_file_path: str, 
                 sim_type: SimulatorType = SimulatorType.WNTRS,
                 seed: Optional[int] = None,
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
        # Initialize base simulator without leaks
        self.simulator_without_leaks = WaterNetworksimulator(inp_file_path, sim_type, verbose)
        
        # Initialize attributes
        self.inp_file_path = inp_file_path
        self.sim_type = sim_type
        self.verbose = verbose
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create a separate simulator instance for the network with leaks
        self.simulator_with_leaks = None
        
        # Store results from both simulations
        self.baseline_results = None
        self.leak_results = None
        
        # Track the added leaks
        self.leak_info = {}
        
    def setup_baseline_simulation(self, 
                                 random_demand: bool = True,
                                 prob_noise: float = 0.1,
                                 min_demand: float = 0.001,
                                 max_demand: float = 0.005,
                                 pattern_hours: int = 6,
                                 tank_fill_percent: float = 75) -> None:
        """
        Set up the baseline simulation without leaks.
        
        Parameters:
        -----------
        random_demand : bool, optional
            Whether to add random demand noise
        prob_noise : float, optional
            Probability of adding noise to a junction
        min_demand : float, optional
            Minimum demand value (m³/s)
        max_demand : float, optional
            Maximum demand value (m³/s)
        pattern_hours : int, optional
            Duration of the demand pattern in hours
        tank_fill_percent : float, optional
            Percentage of tank capacity to fill
        """
        # Set tank levels
        self.simulator_without_leaks.set_tank_levels(fill_percent=tank_fill_percent)
        
        # Add random demand noise if requested
        if random_demand:
            self.simulator_without_leaks.add_random_demand_noise(
                prob_noise=prob_noise,
                min_demand=min_demand,
                max_demand=max_demand
            )
            
            # Add nighttime pattern
            self.simulator_without_leaks.add_nighttime_pattern(pattern_hours=pattern_hours)
        
        # Set simulation duration
        self.simulator_without_leaks.wn.options.time.duration = pattern_hours * 3600  # seconds
        
        if self.verbose:
            print("Baseline simulation setup complete")
    
    def run_baseline_simulation(self) -> None:
        """
        Run the baseline simulation without leaks.
        """
        # Run the simulation
        self.simulator_without_leaks.run_simulation()
        
        # Store results
        self.baseline_results = self.simulator_without_leaks.results
        
        if self.verbose:
            print("Baseline simulation completed")
    
    def setup_leak_simulation(self, 
                             leak_percent: float = 0.1, 
                             start_time: int = 0,
                             duration: Optional[int] = None) -> Dict:
        """
        Set up the leak simulation with specified percentage of pipes leaking.
        
        Parameters:
        -----------
        leak_percent : float, optional
            Percentage of pipes to add leaks to (0-1)
        start_time : int, optional
            Time to start the leak in seconds
        duration : int, optional
            Duration of the leak in seconds. If None, leak persists until end of simulation.
            
        Returns:
        --------
        Dict
            Information about added leaks
        """
        # Create a copy of the baseline simulator
        self.simulator_with_leaks = WaterNetworksimulator(
            self.inp_file_path, 
            self.sim_type,
            self.verbose
        )
        
        # Apply the same configuration as the baseline
        if hasattr(self.simulator_without_leaks.wn, 'options'):
            # Copy duration
            self.simulator_with_leaks.wn.options.time.duration = \
                self.simulator_without_leaks.wn.options.time.duration
                
            # Copy other relevant options if needed
            
        # Initialize leak generator
        leak_generator = LeakGenerator(self.simulator_with_leaks.wn, self.seed)
        
        # Add leaks to the network
        self.leak_info = leak_generator.add_leaks(
            leak_percent=leak_percent,
            start_time=start_time,
            duration=duration
        )
        
        if self.verbose:
            print(f"Added {len(self.leak_info)} leaks to the network")
            
        return self.leak_info
        
    def add_specific_leak(self, 
                         pipe_name: str, 
                         severity: Union[str, LeakSeverity] = None,
                         start_time: int = 0,
                         duration: Optional[int] = None) -> Dict:
        """
        Add a leak to a specific pipe in the network.
        
        Parameters:
        -----------
        pipe_name : str
            Name of the pipe to add leak to
        severity : str or LeakSeverity, optional
            Severity of the leak. Can be 'small', 'medium', 'large', or 'burst'.
            If None, a random severity is chosen.
        start_time : int, optional
            Time to start the leak in seconds
        duration : int, optional
            Duration of the leak in seconds
            
        Returns:
        --------
        Dict
            Information about the added leak
        """
        # Create simulator with leaks if not exists
        if self.simulator_with_leaks is None:
            self.simulator_with_leaks = WaterNetworksimulator(
                self.inp_file_path, 
                self.sim_type,
                self.verbose
            )
            
            # Apply the same configuration as the baseline
            if hasattr(self.simulator_without_leaks.wn, 'options'):
                # Copy duration
                self.simulator_with_leaks.wn.options.time.duration = \
                    self.simulator_without_leaks.wn.options.time.duration
        
        # Initialize leak generator
        leak_generator = LeakGenerator(self.simulator_with_leaks.wn, self.seed)
        
        # Convert string severity to enum if needed
        if severity is not None and isinstance(severity, str):
            severity = LeakSeverity[severity.upper()]
        
        # Add leak to specific pipe
        new_leak_info = leak_generator.add_targeted_leak(
            pipe_name=pipe_name,
            severity=severity,
            start_time=start_time,
            duration=duration
        )
        
        # Update leak info
        self.leak_info.update(new_leak_info)
        
        if self.verbose:
            print(f"Added leak to pipe {pipe_name}")
            
        return new_leak_info
    
    def run_leak_simulation(self) -> None:
        """
        Run the simulation with leaks.
        """
        if self.simulator_with_leaks is None:
            raise ValueError("Leak simulation not set up. Call setup_leak_simulation() first.")
        
        # Run the simulation
        self.simulator_with_leaks.run_simulation()
        
        # Store results
        self.leak_results = self.simulator_with_leaks.results
        
        if self.verbose:
            print("Leak simulation completed")
    
    def run_scenarios(self, 
                     leak_percent: float = 0.1, 
                     start_time: int = 0,
                     duration: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Run both baseline and leak scenarios in sequence.
        
        Parameters:
        -----------
        leak_percent : float, optional
            Percentage of pipes to add leaks to (0-1)
        start_time : int, optional
            Time to start the leak in seconds
        duration : int, optional
            Duration of the leak in seconds
            
        Returns:
        --------
        Tuple[Dict, Dict]
            Baseline and leak simulation results
        """
        # Run baseline simulation if not already run
        if self.baseline_results is None:
            self.run_baseline_simulation()
        
        # Setup leak simulation
        self.setup_leak_simulation(
            leak_percent=leak_percent,
            start_time=start_time,
            duration=duration
        )
        
        # Run leak simulation
        self.run_leak_simulation()
        
        return {
            'baseline': self.baseline_results,
            'leak': self.leak_results,
            'leak_info': self.leak_info
        }
    
    def get_pressure_differences(self) -> Dict:
        """
        Calculate pressure differences between baseline and leak scenarios.
        
        Returns:
        --------
        Dict
            Dictionary with pressure differences and statistics
        """
        if self.baseline_results is None or self.leak_results is None:
            raise ValueError("Both baseline and leak simulations must be run first.")
        
        # Get pressure data
        baseline_pressures = self.baseline_results.node['pressure']
        leak_pressures = self.leak_results.node['pressure']
        
        # Calculate differences
        pressure_diff = baseline_pressures - leak_pressures
        
        # Calculate statistics
        pressure_stats = {
            'mean_diff': pressure_diff.mean(),
            'max_diff': pressure_diff.max(),
            'min_diff': pressure_diff.min(),
            'std_diff': pressure_diff.std(),
            'pressure_diff': pressure_diff,
            'nodes_with_max_diff': pressure_diff.idxmax(),
            'time_of_max_diff': pressure_diff.max(axis=1).idxmax(),
        }
        
        return pressure_stats
    
    def get_flow_differences(self) -> Dict:
        """
        Calculate flow differences between baseline and leak scenarios.
        
        Returns:
        --------
        Dict
            Dictionary with flow differences and statistics
        """
        if self.baseline_results is None or self.leak_results is None:
            raise ValueError("Both baseline and leak simulations must be run first.")
        
        # Get flow data
        baseline_flows = self.baseline_results.link['flowrate']
        leak_flows = self.leak_results.link['flowrate']
        
        # Calculate differences
        flow_diff = baseline_flows - leak_flows
        
        # Calculate statistics
        flow_stats = {
            'mean_diff': flow_diff.mean(),
            'max_diff': flow_diff.max(),
            'min_diff': flow_diff.min(),
            'std_diff': flow_diff.std(),
            'flow_diff': flow_diff,
            'links_with_max_diff': flow_diff.idxmax(),
            'time_of_max_diff': flow_diff.max(axis=1).idxmax(),
        }
        
        return flow_stats
    
    def get_leak_flow_rates(self) -> Dict:
        """
        Calculate the total volume and rate of water lost through leaks.
        
        Returns:
        --------
        Dict
            Dictionary with leak flow rates and total volume
        """
        if self.leak_results is None:
            raise ValueError("Leak simulation must be run first.")
        
        # Get demand data from leak nodes
        leak_flows = {}
        total_leak_volume = 0.0
        
        for leak_name in self.leak_info.keys():
            if leak_name in self.leak_results.node['demand'].columns:
                # Get demand time series for this leak
                leak_demand = self.leak_results.node['demand'][leak_name]
                
                # Store in results
                leak_flows[leak_name] = leak_demand
                
                # Calculate total volume (m³)
                # Need to convert from flow rate to volume
                time_step = self.simulator_with_leaks.wn.options.time.report_timestep
                total_leak_volume += leak_demand.sum() * time_step
        
        # Get simulation duration in hours
        duration_hours = self.simulator_with_leaks.wn.options.time.duration / 3600
        
        # Calculate average leak rate
        avg_leak_rate = total_leak_volume / duration_hours if duration_hours > 0 else 0
        
        return {
            'leak_flows': leak_flows,
            'total_leak_volume_m3': total_leak_volume,
            'avg_leak_rate_m3_per_hour': avg_leak_rate,
            'leak_info': self.leak_info
        }
    
    def get_pressure_sensitivity(self) -> Dict:
        """
        Calculate pressure sensitivity to leaks for each node.
        
        Returns:
        --------
        Dict
            Dictionary with pressure sensitivity metrics
        """
        if self.baseline_results is None or self.leak_results is None:
            raise ValueError("Both baseline and leak simulations must be run first.")
        
        # Get pressure data
        baseline_pressures = self.baseline_results.node['pressure']
        leak_pressures = self.leak_results.node['pressure']
        
        # Calculate absolute differences
        pressure_diff = (baseline_pressures - leak_pressures).abs()
        
        # Calculate mean and max difference for each node
        mean_diff = pressure_diff.mean()
        max_diff = pressure_diff.max()
        
        # Identify most sensitive nodes
        most_sensitive_nodes = mean_diff.sort_values(ascending=False).head(10)
        
        return {
            'mean_sensitivity': mean_diff,
            'max_sensitivity': max_diff,
            'most_sensitive_nodes': most_sensitive_nodes,
            'sensitivity_matrix': pressure_diff
        }
    
    def get_network_resilience_metrics(self) -> Dict:
        """
        Calculate network resilience metrics under leak conditions.
        
        Returns:
        --------
        Dict
            Dictionary with resilience metrics
        """
        if self.baseline_results is None or self.leak_results is None:
            raise ValueError("Both baseline and leak simulations must be run first.")
        
        # Get pressure data
        baseline_pressures = self.baseline_results.node['pressure']
        leak_pressures = self.leak_results.node['pressure']
        
        # Calculate percentage of nodes with pressure below threshold
        pressure_threshold = 20  # meters - typical minimum service pressure
        nodes_below_threshold_baseline = (baseline_pressures < pressure_threshold).sum(axis=1)
        nodes_below_threshold_leak = (leak_pressures < pressure_threshold).sum(axis=1)
        
        # Total number of nodes
        total_nodes = baseline_pressures.shape[1]
        
        # Calculate percentage
        percent_affected_baseline = nodes_below_threshold_baseline / total_nodes * 100
        percent_affected_leak = nodes_below_threshold_leak / total_nodes * 100
        
        # Calculate resilience metrics
        resilience_metrics = {
            'pressure_threshold': pressure_threshold,
            'affected_nodes_baseline': nodes_below_threshold_baseline,
            'affected_nodes_leak': nodes_below_threshold_leak,
            'percent_affected_baseline': percent_affected_baseline,
            'percent_affected_leak': percent_affected_leak,
            'resilience_impact': percent_affected_leak - percent_affected_baseline,
            'total_nodes': total_nodes
        }
        
        return resilience_metrics
    
    