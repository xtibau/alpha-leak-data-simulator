from water_netowrk_simulator.leak_simulator import LeakSimulator
from water_netowrk_simulator.models import Config


class DatasetGenerator:
    def __init__(self, simulator: LeakSimulator, config: Config) -> None:
        """
        Initialize the dataset generator.
        
        Parameters:
        -----------
        simulator : LeakSimulator
            The leak simulator instance
        config : Config
            Configuration object for the dataset generation
        """
        self.simulator = simulator
        self.config = config
    