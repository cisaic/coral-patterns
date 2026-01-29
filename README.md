# Course project for Complex Systems Simulations

## SETUP

1. **Clone the repository (if you have not already):**
   ```bash
   git clone <repository-ssh>
   cd <repository-folder>
   ```

2. **Create & activate your conda environment:**
   ```bash
   conda create -n coral-env python=3.10
   conda activate coral-env
   ```

3. **Install the package and its dependencies:**
   ```bash
    pip install -e .
   ```

   This will install all dependencies listed in `setup.py` automatically.

## Project structure

Directories:

- **coral-patterns/**  
  This is the package directory that contains all the source code for the simulations, model implementation, and utilities. 

- **scripts/**  
  Scripts for analysis, experimentation, and visualization of project results. Imports functions from the coral-patterns package.

- **data/**  
  Contains datasets or saved simulation output and results.

- **tests/**  
  Includes test scripts for validating and verifying the models and functions.

- **plots/**
  Experiment outputs

- **images/**
  Images for animations showing how:
  - Baseline DLA works
  - Growth mode parameter modifies attachment probability in the DLA
  - Friendliness parameter modifies attachment probability in the DLA

Files:

- **setup.py**  
  Configures the package and its dependencies. Allows for easy reproduction of the development environment. Please add any new dependencies you install to this file so we are all working with the same environment.

- **genai-usage.md**  
  Documents generative AI tool usage in the project.

## USAGE
- **scripts/01-run-dla.py**

- **scripts/02-multifractality-experiment.py**
Demonstrates that the DLA model exhibits multifractality, as described by (Halsey TC. 2000.)


# References 
[1] Llabrés E, Re E, Pluma N, Sintes T, Duarte CM. 2024. A generalized numerical model for clonal growth in scleractinian coral colonies. Proceedings. Biological sciences. 291(2030):20241327. doi:10.1098/rspb.2024.1327. http://dx.doi.org/10.1098/rspb.2024.1327.​

[2] Halsey TC. 2000. Diffusion-Limited Aggregation: A Model for Pattern Formation. Physics Today. 53(11):36–41. doi:10.1063/1.1333284. http://dx.doi.org/10.1063/1.1333284.
