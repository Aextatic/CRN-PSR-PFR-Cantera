import cantera as ct
from pathlib import Path
import numpy as np
import json

def calculate_thermal_diffusivity(gas):
    k = gas.thermal_conductivity    # [W/m-K]
    rho = gas.density              # [kg/m^3]
    cp = gas.cp_mass              # [J/kg-K]
    return k / (rho * cp)         # [m^2/s]

def calculate_flame_speed(gas, width=0.03, loglevel=1):
    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    
    # Solve with multicomponent transport model
    f.transport_model = 'multicomponent'
    f.solve(loglevel=loglevel, auto=True)
    
    return f.velocity[0]  # [m/s]

def calculate_flame_thickness(phi):
    # Initial conditions post-compression
    T = 412.04
    P = 253312.4
    FUEL = 'H2:1.0'
    OXIDIZER = 'O2:0.21, N2:0.79'
    
    # Create Cantera gas object 
    gas = ct.Solution("gri30.yaml")
    gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
    gas.TP = T, P
    
    # Calculate thermal diffusivity
    alpha = calculate_thermal_diffusivity(gas)
    
    # Calculate flame speed
    S_L = calculate_flame_speed(gas)
    
    # Calculate flame thickness
    thickness = alpha / S_L
    
    return alpha, S_L, thickness

# Calculate for both equivalence ratios
phi_values = [0.35, 1.2]  # lean and rich conditions

for phi in phi_values:
    alpha, S_L, thickness = calculate_flame_thickness(phi)
    print(f"\nResults for phi = {phi}:")
    print(f"Thermal diffusivity: {alpha:.6e} m²/s")
    print(f"Flame speed: {S_L:.6f} m/s")
    print(f"Flame thickness: {thickness:.6e} m")

# Dictionary to store results
results = {
    "data": []
}

for phi in phi_values:
    alpha, S_L, thickness = calculate_flame_thickness(phi)
    print(f"\nResults for phi = {phi}:")
    print(f"Thermal diffusivity: {alpha:.6e} m²/s")
    print(f"Flame speed: {S_L:.6f} m/s")
    print(f"Flame thickness: {thickness:.6e} m")
    
    # Store results in dictionary
    results["data"].append({
        "Phi": phi,
        "Thermal Diffusivity": alpha,
        "Flame Speed": S_L,
        "Flame Thickness": thickness
    })

# Save results to JSON file
output_path = Path(r"C:\Users\Adnane\OneDrive\Documents\Study\ENSMA\AME EPROP\Y2\BE\Code\Results\flame_results.json")

with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)
