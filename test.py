# Example usage
import wntr
from water_netowrk_simulator.simulator import WaterNetworksimulator
from water_netowrk_simulator.utils.plots import plot_junction_pressure_vs_elevation, debug_node_name_matching


if __name__ == "__main__":
    
    regular_demand = False

    input_file_path = '/workspaces/alpha-leak-data-simulator/inp_files/E2_PLOS1.inp'
    wn = wntr.network.WaterNetworkModel(input_file_path)
    
    if regular_demand:


        # pressure_results = res.node['pressure']
        time_options = wn.options.time
        pattern_timestep = time_options.pattern_timestep
        duration = time_options.duration
        num_steps = int(duration / pattern_timestep) + 1  # +1 para incluir el tiempo 0

        # Crear un patrón de ceros
        zero_pattern_name = 'ZERO_PATTERN'
        zero_multipliers = [0.0] * num_steps
        wn.add_pattern(zero_pattern_name, zero_multipliers)

        # Aplicar el patrón de ceros a todos los nodos con demanda
        for junction_name, junction in wn.junctions():
            # Si el nodo tiene demandas, asignamos el patrón de ceros a cada una
            if junction.demand_timeseries_list:
                for demand in junction.demand_timeseries_list:
                    demand.pattern_name = zero_pattern_name
            else:
                # Si el nodo no tiene demandas, podemos agregar una con valor base 0
                # (aunque técnicamente no es necesario ya que 0 * cualquier patrón = 0)
                junction.add_demand(0.0, zero_pattern_name)

        # wn.patterns.clear()
        sim = wntr.sim.EpanetSimulator(wn)
        res = sim.run_sim()
        res.node['pressure']
    else:
    
        simulator = WaterNetworksimulator(input_file_path)

        # Set tank levels to 75% of their capacity
        simulator.set_tank_levels(fill_percent=75)

        # Assign rnandom demand to all junctions
        simulator.add_random_demand_noise(min_demand=0.001, max_demand=0.005)

        # Add nighttime pattern
        simulator.add_nighttime_pattern()

        # Simulation for 6 hours
        simulator.wn.options.time.duration = 6 * 3600  # en segundos

        # Run it
        simulator.run_simulation()

        # Results
        pressures = simulator.results.node['pressure']

        plot_junction_pressure_vs_elevation(pressures, simulator.wn)