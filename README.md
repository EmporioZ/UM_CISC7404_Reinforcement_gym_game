# UM_CISC7404_Reinforcement_gym_game

This repository contains the code and resources for the CISC7404 course project, focusing on reinforcement learning using Gymnax-style environments.

## Project Description

Soft actor-critic Discrete implementation by Jax and Equinox

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/EmporioZ/UM_CISC7404_Reinforcement_gym_game.git
   cd UM_CISC7404_Reinforcement_gym_game
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n env_name
   conda activate env_name  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Directory
- skittles.py *Skittles* game
- sac-discrete-skittles.py *soft actor-critic discrete* implementation for skittles

## Usage 

To train the agent, run the following command:
```bash
python sac-discrete-skittles.py
```
