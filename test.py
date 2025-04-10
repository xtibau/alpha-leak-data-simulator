# Example usage
from water_netowrk_simulator.simulator import PressureSimulator

if __name__ == "__main__":
    # Initialize the simulator with your INP file
    simulator = PressureSimulator('inp_files/CTOWN.INP')
    
    # Simulate nighttime conditions with tanks at 80% capacity
    pressures = simulator.simulate_night_conditions(tank_fill_percent=80)
    
    # Print junction pressures
    print("Junction Pressures:")
    print(pressures)
    
    # Calculate and print link pressures
    link_pressures = simulator.get_link_pressures(pressures)
    print("\nLink Pressures (estimated):")
    for link, pressure in list(link_pressures.items())[:5]:  # Show first 5 as example
        print(f"{link}: {pressure:.2f}")