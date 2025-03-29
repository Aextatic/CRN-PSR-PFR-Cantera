**# Combustion Simulation Project**
# Overview
This project contains a suite of Python scripts for simulating various combustion phenomena using the Cantera library. The simulations cover compressor performance, flame characteristics, and reactor dynamics—including both lean and rich combustion scenarios—using perfectly stirred reactor (PSR) and plug flow reactor (PFR) models. The project outputs include JSON files with simulation results, reaction path diagrams, and plots for analyzing key parameters such as temperature profiles and pollutant formation.

# File Structure
## Compressor.py
Simulates compressor exit conditions by computing post-compression temperature and pressure based on an isentropic efficiency model. The simulation results are saved to a JSON file.

## diffusivity_flameSpeed.py
Calculates thermal diffusivity, flame speed, and flame thickness for specified equivalence ratios (lean and rich conditions). The script prints the results and stores them in a JSON file.

## PSR_lean.py
Performs lean combustion simulations using a perfectly stirred reactor (PSR) model. It iterates over decreasing residence times to capture reactor dynamics, outputting temperature, pressure, species mole fractions, and heat release rates.

## PSR_rich.py
Conducts rich combustion simulations using a PSR model. Similar to the lean simulation, it records reactor behavior under rich fuel-air mixtures and saves the data to a JSON file.

## PSR_PFR_network.py
Integrates the results from the lean and rich PSR simulations into a reactor network. This script models mixing, dilution, and subsequent plug flow reactor (PFR) segments to analyze temperature evolution and pollutant (NO, NO₂) formation. It also generates reaction path diagrams (DOT files) and plots for temperature and species profiles along the reactor.

# Requirements
Python 3.x

Cantera – for chemical kinetics and reactor simulations.

NumPy – for numerical operations.

Matplotlib – for plotting simulation results.

Pandas – (used in PSR_PFR_network.py for data handling).

Standard libraries such as json and pathlib.

# Installation
Set up a virtual environment (optional but recommended):

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required packages:
pip install cantera numpy matplotlib pandas json os

# JSON Files:
Each simulation saves its results in JSON format (e.g., COMP_results.json, PSR_lean_results.json, PSR_rich_results.json) to a specified results directory.

# Plots:
The scripts generate plots showing temperature profiles, mass flow rate evolution, and pollutant formation along the reactor length.

# Reaction Path Diagrams:
DOT files (e.g., path1.dot, path2.dot, etc.) are generated to illustrate the reaction pathways during the mixing and reactor operations.

# Customization
## Simulation Parameters:
Parameters such as equivalence ratios, residence times, and mass flow rates are defined within the scripts and can be adjusted to explore different combustion conditions.

## Output Paths:
Modify the directory paths in the scripts if you wish to change the location where results and diagrams are saved.

# License
This project is provided for educational and research purposes. You are free to modify and distribute the code as needed.

# Contact
For any questions, feedback, or collaboration please contact Adnane Ait Zidane at adnane-ait.zidane@etu.isae-ensma.fr