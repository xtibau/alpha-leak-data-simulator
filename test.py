# Example usage
import wntr
from water_netowrk_simulator.simulator import WaterNetworksimulator
from water_netowrk_simulator.utils.plots import plot_junction_pressure_vs_elevation, debug_node_name_matching


if __name__ == "__main__":
    

    input_file_path = '/workspaces/alpha-leak-data-simulator/inp_files/E2_PLOS1.inp'
    wn = wntr.network.WaterNetworkModel(input_file_path)

    
    simulator = WaterNetworksimulator(input_file_path)

    # Assign rnandom demand to all junctions
    simulator.add_random_demand_noise(prob_noise=0.1,
                                        min_demand=0.001, 
                                        max_demand=0.005)

    # Add nighttime pattern
    simulator.add_nighttime_pattern()

    # Simulation for 6 hours
    simulator.wn.options.time.duration = 6 * 3600  # en segundos

    # Run it
    simulator.run_simulation()

    # Results
    pressures = simulator.results.node['pressure']

    plot_junction_pressure_vs_elevation(pressures, simulator.wn)