import os, json
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import pandas as pd

# ------------------------------------------------------------------------------
# Load PSR Lean results from JSON file
# ------------------------------------------------------------------------------
RESULTS_DIR = r'C:\\Users\Adnane\\OneDrive\\Documents\\Study\\ENSMA\\AME EPROP\\Y2\\BE\\Code\\Results'
filename = "PSR_lean_results.json"
json_file_path = os.path.join(RESULTS_DIR, filename)

print("----------------------------------------\n")
print(f"[INFO] Loading PSR Lean results from: {json_file_path}")

with open(json_file_path, 'r') as f:
    results = json.load(f)

# ------------------------------------------------------------------------------
# Extract the PSR Lean state at a chosen residence time
# ------------------------------------------------------------------------------
desired_residence_time_lean = 0.0102
RESIDENCE_TIME = np.array(results["Residence time"])

print(f"[INFO] Desired residence time = {desired_residence_time_lean} s")
print(f"[INFO] Checking if this residence time is in the results...")

if desired_residence_time_lean in RESIDENCE_TIME:
    index = np.where(RESIDENCE_TIME == desired_residence_time_lean)[0][0]
    print(f"[SUCCESS] Found exact index for residence time at {index}")
else:
    raise ValueError(f"[ERROR] Residence time {desired_residence_time_lean} not found in JSON results.")

# ------------------------------------------------------------------------------
# Extract T, P, and mole fractions for the exact match
# ------------------------------------------------------------------------------
T_psr_lean = results["Temperature"][index]
P_psr_lean = results["Pressure"][index]
mole_fractions_lean = {
    species: results["Species mole fractions"][species][index] 
    for species in results["Species mole fractions"] 
    if results["Species mole fractions"][species]
}

print("[INFO] Raw sum of extracted mole fractions before any renormalization:")
raw_sum = sum(mole_fractions_lean.values())
print(f"Sum = {raw_sum:.6f}")

# Normalize if needed (only if sum < 1)
# (We assume we want to scale up to 1 if slightly under due to numerical truncation)
if raw_sum < 1.0:
    mole_fractions_lean = {
        species: mf / raw_sum for species, mf in mole_fractions_lean.items()
    }

final_sum_mole_fractions_lean = sum(mole_fractions_lean.values())
print(f"[INFO] Final sum of mole fractions for fuel-lean PSR after potential normalization = {final_sum_mole_fractions_lean:.6f}")
print("----------------------------------------\n")

# ------------------------------------------------------------------------------
# Load PSR Rich results from JSON file
# ------------------------------------------------------------------------------
RESULTS_DIR = r'C:\\Users\Adnane\\OneDrive\\Documents\\Study\\ENSMA\\AME EPROP\\Y2\\BE\\Code\\Results'
filename = "PSR_rich_results.json"
json_file_path = os.path.join(RESULTS_DIR, filename)

print(f"[INFO] Loading PSR Rich results from: {json_file_path}")

with open(json_file_path, 'r') as f:
    results = json.load(f)

# ------------------------------------------------------------------------------
# Extract the PSR Rich state at a chosen residence time
# ------------------------------------------------------------------------------
desired_residence_time_rich = 0.0001
RESIDENCE_TIME = np.array(results["Residence time"])

print(f"[INFO] Desired residence time = {desired_residence_time_rich} s")
print(f"[INFO] Checking if this residence time is in the results...")

if desired_residence_time_rich in RESIDENCE_TIME:
    index = np.where(RESIDENCE_TIME == desired_residence_time_rich)[0][0]
    print(f"[SUCCESS] Found exact index for residence time at {index}")
else:
    raise ValueError(f"[ERROR] Residence time {desired_residence_time_rich} not found in JSON results.")

# ------------------------------------------------------------------------------
# Extract T, P, and mole fractions for the exact match
# ------------------------------------------------------------------------------
T_psr_rich = results["Temperature"][index]
P_psr_rich = results["Pressure"][index]
mole_fractions_rich = {
    species: results["Species mole fractions"][species][index] 
    for species in results["Species mole fractions"] 
    if results["Species mole fractions"][species]
}

print("[INFO] Raw sum of extracted mole fractions before any renormalization:")
raw_sum = sum(mole_fractions_rich.values())
print(f"Sum = {raw_sum:.6f}")

# Normalize if needed (only if sum < 1)
# (We assume we want to scale up to 1 if slightly under due to numerical truncation)
if raw_sum < 1.0:
    mole_fractions_rich = {
        species: mf / raw_sum for species, mf in mole_fractions_rich.items()
    }

final_sum_mole_fractions_rich = sum(mole_fractions_rich.values())
print(f"[INFO] Final sum of mole fractions for fuel-rich PSR after potential normalization = {final_sum_mole_fractions_rich:.6f}")
print("----------------------------------------\n")

# ------------------------------------------------------------------------------
# Initialize Cantera gas object with these conditions
# ------------------------------------------------------------------------------
alpha = 0
P_psr = 253312.40  # Pa
mole_fractions_mix = {
    sp: alpha * mole_fractions_lean.get(sp, 0.0) + (1 - alpha) * mole_fractions_rich.get(sp, 0.0)
    for sp in set(mole_fractions_lean) | set(mole_fractions_rich)
}
T_psr_mix = alpha * T_psr_lean + (1 - alpha) * T_psr_rich

fuel_species = 'H2'
oxidizer_species = 'O2'

phi_lean = 0.35
phi_rich = 1.2
weight_lean = alpha * mole_fractions_lean[fuel_species]
weight_rich = (1 - alpha) * mole_fractions_rich[fuel_species]

phi_mix = (phi_lean * weight_lean + phi_rich * weight_rich) / (weight_lean + weight_rich)

gas = ct.Solution('gri30.yaml', transport_model='multicomponent')
gas.TPX = T_psr_mix, P_psr, mole_fractions_mix

element = 'N'
ino = gas.species_index('NO')
ino2 = gas.species_index('NO2')

print("[INFO] ----- PSR Conditions Loaded -----")
print(f"        Residence time rich  = {desired_residence_time_rich} s")
print(f"        Residence time lean  = {desired_residence_time_lean} s")
print(f"        PSR1 (lean) phi = {phi_lean:.4f}")
print(f"        PSR2 (rich) phi = {phi_rich:.4f}")
print(f"        Mixed phi (weighted by fuel mole fraction) = {phi_mix:.4f}")
print(f"        Temperature     = {gas.T} K")
print(f"        Pressure        = {gas.P} Pa")
print(f"        NO mass fraction   = {gas.Y[ino]}")
print(f"        NO2 mass fraction  = {gas.Y[ino2]}")
print("----------------------------------------\n")

# ------------------------------------------------------------------------------
# Define input parameters for the subsequent PFR steps
# ------------------------------------------------------------------------------
AIR_MASS_FLOW = 20E-3             # kg/s
HYDROGEN_MASS_FLOW = 0.097E-3     # kg/s
COMBUSTION_AIR_MASS_FLOW_STOI = 3.33E-3   # kg/s stoichiometric air
COMBUSTION_AIR_MASS_FLOW = COMBUSTION_AIR_MASS_FLOW_STOI / phi_mix
PRODUCTS_MASS_FLOW = COMBUSTION_AIR_MASS_FLOW + HYDROGEN_MASS_FLOW
DILUTION_AIR_MASS_FLOW = AIR_MASS_FLOW - COMBUSTION_AIR_MASS_FLOW

print("[INFO] ----- Flow Parameters -----")
print(f"        PHI = {phi_mix}")
print(f"        Total air mass flow       = {AIR_MASS_FLOW} kg/s")
print(f"        Hydrogen mass flow        = {HYDROGEN_MASS_FLOW} kg/s")
print(f"        Stoichiometric air flow   = {COMBUSTION_AIR_MASS_FLOW_STOI} kg/s")
print(f"        PSR combustion air flow   = {COMBUSTION_AIR_MASS_FLOW} kg/s")
print(f"        Products mass flow        = {PRODUCTS_MASS_FLOW} kg/s")
print(f"        Dilution air mass flow    = {DILUTION_AIR_MASS_FLOW} kg/s")
print("----------------------------------\n")

# Initialize tracking arrays for plotting
T_tab = [gas.T, gas.T]
NO_tab = [gas.Y[ino] * PRODUCTS_MASS_FLOW, gas.Y[ino] * PRODUCTS_MASS_FLOW]
NO2_tab = [gas.Y[ino2] * PRODUCTS_MASS_FLOW, gas.Y[ino2] * PRODUCTS_MASS_FLOW]
length_0 = 0.0  # mixing zone offset (m)
x_tab = [0, length_0]

# ------------------------------------------------------------------------------
# First Mixing
# ------------------------------------------------------------------------------
print("[INFO] --- First Mixing Step ---")
A = ct.Quantity(gas, constant='HP')
A.TP = gas.T, gas.P
A.mass = PRODUCTS_MASS_FLOW

print(f"[DEBUG] Stream A (burnt gas) -> T={A.T:.2f} K, P={A.P:.2f} Pa, mass={A.mass:.6e} kg")

B = ct.Quantity(gas, constant='HP')
B.TPX = 412.04, 253312.4, 'O2:0.21, N2:0.79'
B.mass = DILUTION_AIR_MASS_FLOW / 2

print(f"[DEBUG] Stream B (dilution air) -> T={B.T:.2f} K, P={B.P:.2f} Pa, mass={B.mass:.6e} kg")

M_1 = A + B  # Perform the mixing
print("[INFO] Mixed state after first dilution:")
print(f"       T={M_1.T:.2f} K, P={M_1.P:.2f} Pa")

# Update tracking
T_tab.append(M_1.T)
mass_flow_rate_1 = PRODUCTS_MASS_FLOW + DILUTION_AIR_MASS_FLOW / 2
NO_tab.append(M_1.Y[ino] * mass_flow_rate_1)
NO2_tab.append(M_1.Y[ino2] * mass_flow_rate_1)
x_tab.append(length_0)

diagram1 = ct.ReactionPathDiagram(gas, element)
diagram1.show_details = True
diagram1.write_dot('path1.dot')
print("[INFO] Reaction path diagram for first mixing saved as 'path1.dot'\n")

# ------------------------------------------------------------------------------
# First PFR (first segment)
# ------------------------------------------------------------------------------
print("[INFO] --- First PFR Segment ---")
# Replace previous total length with our calculated value for first PFR segment:
pfr_length_1 = 2.13e-2
area = 1.04e-2       
n_steps = 20000

u_1 = mass_flow_rate_1 / (area * M_1.density)
t_total_1 = pfr_length_1 / u_1
dt_1 = t_total_1 / n_steps

print(f"[DEBUG] PFR #1 length = {pfr_length_1:.5e} m, area={area:.5e} m^2")
print(f"[DEBUG] Velocity u_1 = {u_1:.5e} m/s, total time = {t_total_1:.5e} s, dt={dt_1:.5e} s")

# Use the gas composition from M_1 for the reactor initial state
r_1 = ct.IdealGasConstPressureReactor(gas)
sim_1 = ct.ReactorNet([r_1])

t_1_array = (np.arange(n_steps) + 1) * dt_1
x_1 = np.zeros_like(t_1_array)
u_1_array = np.zeros_like(t_1_array)
states_1 = ct.SolutionArray(r_1.thermo)

for i, t_i in enumerate(t_1_array):
    sim_1.advance(t_i)
    u_1_array[i] = mass_flow_rate_1 / (area * r_1.thermo.density)
    x_1[i] = x_1[i - 1] + u_1_array[i] * dt_1
    states_1.append(r_1.thermo.state)

    T_tab.append(r_1.thermo.T)
    NO_tab.append(r_1.thermo.Y[ino] * mass_flow_rate_1)
    NO2_tab.append(r_1.thermo.Y[ino2] * mass_flow_rate_1)
    x_tab.append(x_1[i] + length_0)

diagram2 = ct.ReactionPathDiagram(gas, element)
diagram2.show_details = True
diagram2.write_dot('path2.dot')
print(f"[INFO] End of First PFR: final temperature = {T_tab[-1]:.2f} K")
print("[INFO] Reaction path diagram for first PFR saved as 'path2.dot'\n")

# ------------------------------------------------------------------------------
# Second Mixing
# ------------------------------------------------------------------------------
print("[INFO] --- Second Mixing Step ---")
C = ct.Quantity(gas, constant='HP')
C.TP = r_1.thermo.T, r_1.thermo.P
C.mass = mass_flow_rate_1

print(f"[DEBUG] Stream C (exhaust from first PFR) -> T={C.T:.2f} K, P={C.P:.2f} Pa, mass={C.mass:.6e} kg")

B = ct.Quantity(gas, constant='HP')
B.TPX = 412.04, 253312.4, 'O2:0.21, N2:0.79'
B.mass = DILUTION_AIR_MASS_FLOW / 2

print(f"[DEBUG] Stream B (dilution air) -> T={B.T:.2f} K, P={B.P:.2f} Pa, mass={B.mass:.6e} kg")

M_2 = C + B
print("[INFO] Mixed state after second dilution:")
print(f"       T={M_2.T:.2f} K, P={M_2.P:.2f} Pa")

T_tab.append(M_2.T)
mass_flow_rate_2 = mass_flow_rate_1 + DILUTION_AIR_MASS_FLOW / 2
NO_tab.append(M_2.Y[ino] * mass_flow_rate_2)
NO2_tab.append(M_2.Y[ino2] * mass_flow_rate_2)
x_tab.append(x_1[-1] + length_0)

diagram3 = ct.ReactionPathDiagram(gas, element)
diagram3.show_details = True
diagram3.write_dot('path3.dot')
print("[INFO] Reaction path diagram for second mixing saved as 'path3.dot'\n")

# ------------------------------------------------------------------------------
# Second PFR (second segment)
# ------------------------------------------------------------------------------
print("[INFO] --- Second PFR Segment ---")
# Use our calculated value for the second PFR segment:
pfr_length_2 = 1.07e-2  # 0.012 cm converted to meters
u_2 = mass_flow_rate_2 / (area * M_2.density)
t_total_2 = pfr_length_2 / u_2
dt_2 = t_total_2 / n_steps

print(f"[DEBUG] PFR #2 length = {pfr_length_2:.5e} m, area={area:.5e} m^2")
print(f"[DEBUG] Velocity u_2 = {u_2:.5e} m/s, total time = {t_total_2:.5e} s, dt={dt_2:.5e} s")

r_2 = ct.IdealGasConstPressureReactor(gas)
sim_2 = ct.ReactorNet([r_2])

t_2_array = (np.arange(n_steps) + 1) * dt_2
x_2 = np.zeros_like(t_2_array)
u_2_array = np.zeros_like(t_2_array)
states_2 = ct.SolutionArray(r_2.thermo)

for i, t_i in enumerate(t_2_array):
    sim_2.advance(t_i)
    u_2_array[i] = mass_flow_rate_2 / (area * r_2.thermo.density)
    x_2[i] = x_2[i - 1] + u_2_array[i] * dt_2
    states_2.append(r_2.thermo.state)
    
    T_tab.append(r_2.thermo.T)
    NO_tab.append(r_2.thermo.Y[ino] * mass_flow_rate_2)
    NO2_tab.append(r_2.thermo.Y[ino2] * mass_flow_rate_2)
    x_tab.append(x_1[-1] + length_0 + x_2[i])

diagram4 = ct.ReactionPathDiagram(gas, element)
diagram4.show_details = True
diagram4.write_dot('path4.dot')

print(f"[INFO] End of Second PFR: final temperature = {T_tab[-1]:.2f} K")
print("[INFO] Reaction path diagram for second PFR saved as 'path4.dot'")
print("[INFO] NO flow at exit = {:.4e} kg/s".format(NO_tab[-1]))
print("[INFO] NO2 flow at exit = {:.4e} kg/s".format(NO2_tab[-1]))
print("----------------------------------------\n")

# ------------------------------------------------------------------------------
# Plot Results
# ------------------------------------------------------------------------------
print("[INFO] Plotting results...")

plt.figure()
plt.plot(np.array(x_tab) * 1000, T_tab)
plt.xlabel('$x$ [mm]')
plt.ylabel('$T$ [K]')
plt.title('Temperature Profile Along Reactor')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.array(x_tab) * 1000, np.array(NO_tab) * 1000)
plt.xlabel('$x$ [mm]')
plt.ylabel('$NO$ [g/s]')
plt.title('NO Mass vs. Position')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(np.array(x_tab) * 1000, np.array(NO2_tab) * 1000)
plt.xlabel('$x$ [mm]')
plt.ylabel('$NO2$ [g/s]')
plt.title('NO2 Mass vs. Position')
plt.grid(True)
plt.show()

print("[INFO] Simulation and plotting complete.")

