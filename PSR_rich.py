import os, json
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

# ------------------------------------------------------------------------------
# Initialize Cantera gas object with these conditions
# ------------------------------------------------------------------------------
gas = ct.Solution('gri30.yaml')
PHI = 1.2  # rich combustion
gas.TP = 412.04, 253312.4
FUEL = 'H2:1.0'
OXIDIZER = 'O2:0.21, N2:0.79'
gas.set_equivalence_ratio(PHI, FUEL, OXIDIZER)

# ------------------------------------------------------------------------------
# Single reactor network setup
# ------------------------------------------------------------------------------
inlet = ct.Reservoir(gas)

gas.equilibrate('HP')
combustor = ct.IdealGasReactor(gas)
combustor.volume = 1.67e-6

exhaust = ct.Reservoir(gas)

def mdot(t):
    return combustor.mass / RESIDENCE_TIME

inlet_mfc = ct.MassFlowController(inlet, combustor, mdot=mdot)

outlet_mfc = ct.PressureController(combustor, exhaust, primary=inlet_mfc, K=0.01)

# the simulation only contains one reactor
sim = ct.ReactorNet([combustor])

# iterate over decreasing residence times
states = ct.SolutionArray(gas, extra=['tres'])
mass_history = []
RESIDENCE_TIME = 0.0001    # set residence time by estimation
while combustor.T > 412.04:
    sim.initial_time = 0.0  # reset the integrator
    sim.advance_to_steady_state()
    mass_history.append(combustor.mass)
    print(f"tres = {RESIDENCE_TIME:.2e}; T = {combustor.T:.1f}; Mass = {combustor.mass:.6e}")
    states.append(combustor.thermo.state, tres=RESIDENCE_TIME)
    RESIDENCE_TIME *= 0.9  # decrease the residence time for the next iteration

# ------------------------------------------------------------------------------
# Create results dictionary
# ------------------------------------------------------------------------------

results = {
    "Residence time": list(states.tres),
    "Density": list(states.density),
    "Mass": list(mass_history),
    "Mass flow rate": list([m / r for m, r in zip(mass_history, states.tres)]),
    "Temperature": list(states.T),
    "Pressure": list(states.P),
    "Species mole fractions": {species: [x for x in states.X[:, i] if x>= 1E-10] for i, species in enumerate(gas.species_names)}
}

RESULTS_DIR = r'C:\\Users\Adnane\\OneDrive\\Documents\\Study\\ENSMA\\AME EPROP\\Y2\\BE\\Code\\Results'
filename = "PSR_rich_results.json"
os.makedirs(RESULTS_DIR, exist_ok=True)
output_file = os.path.join(RESULTS_DIR, filename)
with open(output_file, 'w') as json_file:
    json.dump(results, json_file, indent=4)

# ------------------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------------------

f, ax1 = plt.subplots(1, 1)
ax1.plot(states.tres, states.heat_release_rate, '.-', color='C0')
ax2 = ax1.twinx()
ax2.plot(states.tres[:-1], states.T[:-1], '.-', color='C1')
ax1.set_xlabel('residence time [s]')
ax1.set_ylabel('heat release rate [W/m$^3$]', color='C0')
ax2.set_ylabel('temperature [K]', color='C1')
f.tight_layout()
plt.show()

plt.figure()
plt.plot((results["Residence time"]), results["Mass"]), '-o'
plt.xlabel("Residence Time [s]")
plt.ylabel("Mass Flow Rate [kg/s]")
plt.title("Mass Flow Rate Evolution")
plt.grid()
plt.show()