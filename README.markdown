# UM_CISC7404_Reinforcement_gym_game

This repository contains the code and resources for the CISC7404 course project, focusing on reinforcement learning using OpenAI Gym environments.

## Project Description

The project aims to demonstrate the application of reinforcement learning algorithms to solve game-based problems. It includes the implementation of a custom Gym environment and the training of an agent using a reinforcement learning algorithm.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/EmporioZ/UM_CISC7404_Reinforcement_gym_game.git
   cd UM_CISC7404_Reinforcement_gym_game
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the agent, run the following command:
```bash
python train.py
```

To evaluate the trained agent, run:
```bash
python evaluate.py
```

## Project Structure

- `environments/`: Contains the custom Gym environment definitions.
- `agents/`: Contains the reinforcement learning agent implementations.
- `train.py`: Script to train the agent.
- `evaluate.py`: Script to evaluate the trained agent.
- `utils/`: Utility functions and helpers.
- `models/`: Directory to save trained models.
- `results/`: Directory to save training logs and results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [your_email@example.com].