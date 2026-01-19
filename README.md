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
  This is the package directory that contains all the source code for the simulations, model implementation, and utilities. Anything that can be a function goes here. 

- **scripts/**  
  Scripts for analysis, experimentation, and visualization of project results. Import functions from the coral-patterns package.

- **data/**  
  (If present) Contains datasets or saved simulation output and results.

- **tests/**  
  Includes test scripts for validating and verifying the models and functions.

Files:

- **setup.py**  
  Configures the package and its dependencies. Allows for easy reproduction of the development environment. Please add any new dependencies you install to this file so we are all working with the same environment.

- **genai-usage.md**  
  Documents generative AI tool usage in the project.

# Background
Here we will provide a brief intro to the project

# Model description
Here we will describe our model

# References 
[1] Sample Reference 
