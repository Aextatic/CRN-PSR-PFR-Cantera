import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import os, json

# Constants
AIR_COMPOSITION = "N2:0.79, O2:0.21"
INITIAL_TEMPERATURE = 300.0 
PRESSURE = 101325.0  
PRESSURE_RATIO = 2.5
COMP_ISENTROPIC_EFFICIENCY = 0.9

def initialize_gas():
    """Initialize gas object with given conditions"""
    gas = ct.Solution('gri30.yaml')
    gas.TPX = INITIAL_TEMPERATURE, PRESSURE, AIR_COMPOSITION
    return gas

def calculate_compressor_exit_conditions(gas):
    """Calculate post-compression conditions"""
    P_comp_out = PRESSURE * PRESSURE_RATIO
    gamma = gas.cp / gas.cv
    T_comp_out_isentropic = INITIAL_TEMPERATURE * (P_comp_out / PRESSURE) ** ((gamma - 1) / gamma)
    T_comp_out = INITIAL_TEMPERATURE + (T_comp_out_isentropic - INITIAL_TEMPERATURE) / COMP_ISENTROPIC_EFFICIENCY
    return T_comp_out, P_comp_out

def extract_gas_properties(gas):
    """Extract relevant thermo properties from gas"""
    properties = {
        "T": float(gas.T),
        "P": float(gas.P),
        "composition": {species: float(gas[species].X[0]) for species in gas.species_names},  # Correct access to mole fractions
        "enthalpy": float(gas.enthalpy_mass),
        "entropy": float(gas.entropy_mass),
        "cv": float(gas.cv),
        "cp": float(gas.cp),
        "density": float(gas.density),
    }
    return properties

def save_results_to_json(results, filename="COMP_results.json"):
    """Save results to a JSON file"""
    results_dir = r'C:/Users/Adnane/OneDrive/Documents/Project Files/Python/uGT/Results'
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, filename)
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def main():
    gas = initialize_gas()
    pre_compression_properties = extract_gas_properties(gas)
    T_comp_out, P_comp_out = calculate_compressor_exit_conditions(gas)
    gas.TP = T_comp_out, P_comp_out
    post_compression_properties = extract_gas_properties(gas)

    results = {
        "Pre-Compression": pre_compression_properties,
        "Post-Compression": post_compression_properties
    }
    save_results_to_json(results)

if __name__ == "__main__":
    main()