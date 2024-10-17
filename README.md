Deep Q-Network (DQN) Snake Game AI
This project implements an AI agent that learns to play the classic Snake game using advanced Deep Q-Network (DQN) techniques, including Double DQN, Dueling DQN architecture, and Prioritized Experience Replay. The agent is trained using PyTorch and can leverage multiple GPUs for accelerated training.

Introduction
This project demonstrates how to train an AI agent to play the Snake game using advanced reinforcement learning techniques. The agent uses a Deep Q-Network to learn optimal strategies through experience. The implementation includes:

Double DQN: Reduces overestimation of Q-values.
Dueling DQN Architecture: Separates state value and advantage estimation.
Prioritized Experience Replay: Prioritizes important experiences during training.
Multi-GPU Support: Utilizes multiple GPUs for faster training.

Features
Advanced RL Techniques: Implements Double DQN, Dueling DQN, and Prioritized Experience Replay.
Multi-GPU Training: Supports training on multiple GPUs using PyTorch's DataParallel.
Customizable Hyperparameters: Easy to adjust learning rate, epsilon decay, batch size, etc.
Visualization: Plots training progress and allows you to watch the trained agent play.
Model Saving and Loading: Saves the trained model for future use or retraining.

Prerequisites
Operating System: Windows, macOS, or Linux
Python Version: Python 3.7 or higher (Python 3.8+ recommended)
GPU Support: NVIDIA GPUs with CUDA support (optional but recommended)

Installation
Clone the repository:

python -m venv venv
Windows: source venv/Scripts/activate
macOS/Linux: source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Running the Code
1. Training the Agent

Run either of the three script to start training the agent; read paper comparison.txt comparing the two different type of Neural Network:

python dqn_snake.py --> This will train a model and play the Snake game using an Agent that learns through a Deep Q-Neural Network using Machine Learning.
python doubledqn_snake.py --> This will train a model and play the Snake game using an Agent using a more complex double-deep Q-Neural Network using Machine Learning. 
python qn_snake_game_ai.py ----> This will train a model and play the Snake game using an Agent using a more straightforward, less complex Q-Neural Network, which works well with CPU or GPU. 

Watching the Trained Agent Play
After training, the script will automatically have the agent play a game using the trained model.

To watch the agent play again without retraining, run:
python dqn_snake.py --play

Retraining the Agent
To continue training with the existing model:

python dqn_snake.py

The script will detect the existing model and load it for retraining.

Code Explanation
- Environment Setup
SnakeGame Class: Manages the game environment, including the snake's movement, food placement, collision detection, and UI rendering.
Methods:
reset(): Resets the game to the initial state.
play_step(action): Executes a game step based on the agent's action.
get_state(): Returns the current state representation for the agent.
- Deep Q-Network Agent
DQNAgent Class: Implements the agent interacting with the environment and learning from experiences.
Features:
Neural Network: Uses a Dueling DQN architecture with two hidden layers.
Double DQN Logic: Improves learning stability.
Prioritized Experience Replay: Samples important experiences more frequently.
Multi-GPU Support: Utilizes multiple GPUs for training acceleration.
Methods:
act(state, use_epsilon=True): Determines the action to take based on the current state.
remember(state, action, reward, next_state, done): Stores experiences in replay memory.
replay(): Samples a batch of experiences and performs a learning step.
update_target_model(): Updates the target network with the main network's weights.
save(filename): Saves the trained model to a file.
load(filename): Loads a trained model from a file.
- Training Procedure
train_dqn() Function: Orchestrates the training process.
Initializes the game environment and agent.
Loads an existing model if available.
Runs training episodes, updating the agent's policy.
Saves the trained model after training.
-  Playing with the Trained Agent
play_with_trained_model(env) Function:
Loads the trained model.
Runs a game with the agent using the trained policy.
Displays the game UI to watch the agent play.
Hyperparameters
Key hyperparameters used in the implementation:

Learning Rate: 0.0005
Discount Factor (gamma): 0.99
Exploration Rate (epsilon):
Starts at 1.0 and decays to 0.05 using epsilon_decay = 0.995
Batch Size: 128
Replay Memory Capacity: 200,000
Hidden Layer Size: 512 neurons per layer
Number of Episodes: 10,000 (adjustable)


Troubleshooting:

ModuleNotFoundError: If you encounter this error, ensure all required libraries are installed.
pip install -r requirements.txt

CUDA Errors: If you're experiencing issues with GPU acceleration:
Ensure you have the correct version of CUDA installed.
Verify that your NVIDIA drivers are up to date.
Run a simple PyTorch script to test CUDA availability:

import torch
print(torch.cuda.is_available())

Slow Training: Training can be computationally intensive.
Reduce the number of episodes for testing purposes.
Adjust the batch size or neural network architecture.

Display Issues:
Ensure that Pygame is properly installed.
If the game window doesn't appear, check for any Pygame-related errors in the console.

Acknowledgments
PyTorch: For providing a flexible deep learning framework.
Pygame: For enabling game development and UI rendering.
OpenAI Gym: Inspiration for environment and agent interaction patterns.
Reinforcement Learning Community: For shared knowledge and resources.
