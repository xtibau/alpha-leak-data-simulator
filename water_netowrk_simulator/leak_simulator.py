from simulator import WaterNetworksimulator

class LeakSimulator:
    def __init__(self, inp_file_path: str):
        """
        Initialize the leak generator.
        First performs a simulation to get the initial state of the network.
        It uses the simulator instance to run the simulation.
        Then it generates leaks in the network
        Runs the new simulation.
        
        Parameters:
        -----------
        simulator : WaterNetworksimulator
            Simulator instance to use for simulations
        """
        self.simulator_without_leaks = WaterNetworksimulator(inp_file_path)
        self.simulator_with_leaks = WaterNetworksimulator(inp_file_path)

    """
    We have to create a simulation, normal, then simulate some leaks
    and then simulate the new state of the network.
    The leaks are generated in the simulator_with_leaks instance.
    """
    

    
    