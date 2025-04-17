from simulator import WaterNetworksimulator

class LeakSimulator:
    def __init__(self, simulator: WaterNetworksimulator):
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
        self.simulator = simulator
    
    

    
    