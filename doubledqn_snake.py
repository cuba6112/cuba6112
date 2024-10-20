import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import random
from collections import deque
import pygame
import matplotlib.pyplot as plt

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define constants
WIDTH = 640
HEIGHT = 480
BLOCK_SIZE = 20
SPEED = 100  # Increased speed for faster training

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)   # Red color
GREEN = (0, 255, 0)

# Initialize Pygame
pygame.init()

# Dueling DQN Model with Double DQN logic
class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Dueling streams
        self.fc_value = nn.Linear(hidden_size, 1)
        self.fc_advantage = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Add this new class after the DuelingDQN class
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Snake Game Environment
class SnakeGame:
    def __init__(self, window_id):
        self.window_id = window_id
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f'Snake Game - GPU {window_id}')
        self.clock = pygame.time.Clock()
        self.reset()
        self.display_ui = True

    def reset(self):
        self.direction = 1  # 0: up, 1: right, 2: down, 3: left
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.food = self.place_food()
        self.score = 0
        self.frame_iteration = 0
        return self.get_state()

    def place_food(self):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move snake
        new_head = self.move(action)
        self.snake.insert(0, new_head)

        # Check if game over
        reward = 0.1  # Survival reward
        game_over = False
        if self.is_collision(new_head) or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return self.get_state(), reward, game_over, self.score

        # Check if snake ate food
        if self.snake[0] == self.food:
            self.score += 1
            reward = 10
            self.food = self.place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        if self.display_ui:
            self._update_ui()
            self.clock.tick(SPEED)

        return self.get_state(), reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        if pt[0] >= WIDTH or pt[0] < 0 or pt[1] >= HEIGHT or pt[1] < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        pygame.display.set_caption(f'Snake Game - GPU {self.window_id}')
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        text = pygame.font.Font(None, 36).render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, action):
        # [straight, right, left]
        clock_wise = [0, 1, 2, 3]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.direction == 0:
            y -= BLOCK_SIZE
        elif self.direction == 1:
            x += BLOCK_SIZE
        elif self.direction == 2:
            y += BLOCK_SIZE
        else:
            x -= BLOCK_SIZE

        new_head = (x, y)
        return new_head

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - BLOCK_SIZE, head[1])
        point_r = (head[0] + BLOCK_SIZE, head[1])
        point_u = (head[0], head[1] - BLOCK_SIZE)
        point_d = (head[0], head[1] + BLOCK_SIZE)

        dir_l = self.direction == 3
        dir_r = self.direction == 1
        dir_u = self.direction == 0
        dir_d = self.direction == 2

        # Additional features
        distance_to_food = np.linalg.norm(np.array(self.food) - np.array(head)) / np.sqrt(WIDTH**2 + HEIGHT**2)
        snake_length = len(self.snake) / (WIDTH * HEIGHT / (BLOCK_SIZE * BLOCK_SIZE))

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1],  # food down

            # Additional features
            distance_to_food,
            snake_length
        ]

        return np.array(state, dtype=float)

# Prioritized Experience Replay Memory (Fixed)
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha

    def push(self, error, transition):
        max_priority = max(self.priorities[:len(self.memory)]) if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(max_priority)
        else:
            self.memory[self.pos] = transition
            self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities[:len(self.memory)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        # Importance Sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # small constant to avoid zero priority

# DQN Agent with Double DQN and Prioritized Experience Replay
class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayMemory(200000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.batch_size = 128
        self.beta = 0.4  # For importance-sampling weights in PER
        self.beta_increment_per_sampling = 0.001
        self.device = device
        self.model = DuelingDQN(state_size, 512, action_size).to(device)
        self.target_model = DuelingDQN(state_size, 512, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss(reduction='none')  # For PER

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Compute TD error for priority
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([[action]], dtype=torch.long).to(self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor([done], dtype=torch.float32).to(self.device)

        current_q = self.model(state_tensor).gather(1, action_tensor)
        with torch.no_grad():
            next_action = self.model(next_state_tensor).argmax(1).unsqueeze(1)
            next_q = self.target_model(next_state_tensor).gather(1, next_action)
            target_q = reward_tensor + (1 - done_tensor) * self.gamma * next_q

        td_error = torch.abs(current_q - target_q).item()
        self.memory.push(td_error, (state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory.memory) < self.batch_size:
            return

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        minibatch, indices, weights = self.memory.sample(self.batch_size, self.beta)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q values
        current_q_values = self.model(states).gather(1, actions)

        # Double DQN target Q values
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD errors
        errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors.flatten())

        # Compute loss with PER weights
        loss = (self.criterion(current_q_values, target_q_values) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), filename)
        else:
            torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)
        self.update_target_model()

    def set_device(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.target_model = self.target_model.to(device)

# Modify the train_on_gpu function
def train_on_gpu(gpu_id, shared_model, num_episodes, return_dict):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    env = SnakeGame(gpu_id)
    state_size = 13
    action_size = 3
    agent = DQNAgent(state_size, action_size, device)
    agent.model.load_state_dict(shared_model.state_dict())
    
    for e in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, score = env.play_step(np.eye(action_size)[action])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
            
            # Render the game
            env._update_ui()
            pygame.time.delay(50)  # Adjust delay as needed
        
        if e % 10 == 0:
            agent.update_target_model()
        
        print(f"GPU {gpu_id}, Episode: {e}/{num_episodes}, Score: {score}, Total Reward: {total_reward:.2f}")
    
    # Clone the state dict and move it to CPU before returning
    cpu_state_dict = {k: v.clone().cpu() for k, v in agent.model.state_dict().items()}
    return_dict[gpu_id] = cpu_state_dict

# Modify the train_parallel function
def train_parallel(num_gpus=4, episodes_per_gpu=1200, existing_ensemble=None):
    mp.set_start_method('spawn', force=True)
    pygame.init()
    
    state_size = 13
    action_size = 3
    hidden_size = 512

    if existing_ensemble is None:
        shared_model = DuelingDQN(state_size, hidden_size, action_size)
    else:
        shared_model = existing_ensemble.models[0]  # Use the first model from the ensemble as the shared model
    
    shared_model.share_memory()
    
    processes = []
    return_dict = mp.Manager().dict()
    for gpu_id in range(num_gpus):
        p = mp.Process(target=train_on_gpu, args=(gpu_id, shared_model, episodes_per_gpu, return_dict))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Collect trained models
    trained_models = []
    for i in range(num_gpus):
        if i in return_dict:
            model = DuelingDQN(state_size, hidden_size, action_size)
            model.load_state_dict(return_dict[i])
            trained_models.append(model)
    
    # Ensure we have the correct number of models
    print(f"Number of trained models: {len(trained_models)}")
    
    # Create the ensemble model
    if existing_ensemble is None:
        ensemble_model = EnsembleModel(trained_models)
    else:
        # Update the existing ensemble with new trained models
        ensemble_model = EnsembleModel([existing_ensemble.models[i] if i < len(existing_ensemble.models) else trained_models[i-len(existing_ensemble.models)] for i in range(num_gpus)])
    
    # Save the ensemble model
    save_ensemble_model(ensemble_model, 'snake_agent_ensemble.pth')
    print("Parallel training completed. Ensemble model saved as 'snake_agent_ensemble.pth'")

    return ensemble_model

# Add this new function after the EnsembleModel class
def save_ensemble_model(ensemble_model, filename):
    torch.save({
        'state_dict': ensemble_model.state_dict(),
        'num_models': len(ensemble_model.models),
        'state_size': 13,
        'hidden_size': 512,
        'action_size': 3
    }, filename)

def load_ensemble_model(filename):
    checkpoint = torch.load(filename)
    print(f"Loaded checkpoint keys: {checkpoint.keys()}")
    
    if 'state_dict' in checkpoint:
        # New format
        state_size = checkpoint.get('state_size', 13)
        hidden_size = checkpoint.get('hidden_size', 512)
        action_size = checkpoint.get('action_size', 3)
        num_models = checkpoint.get('num_models', 4)
        print(f"Loading ensemble with {num_models} models")
    else:
        # Old format (assuming it's just the state dict)
        print("Old format detected, using default values")
        state_size, hidden_size, action_size = 13, 512, 3
        num_models = 4  # Assuming 4 GPUs were used
    
    if num_models == 0:
        raise ValueError("Cannot create an ensemble with 0 models")
    
    models = [DuelingDQN(state_size, hidden_size, action_size) for _ in range(num_models)]
    ensemble = EnsembleModel(models)
    
    if 'state_dict' in checkpoint:
        ensemble.load_state_dict(checkpoint['state_dict'])
    else:
        ensemble.load_state_dict(checkpoint)
    
    return ensemble

def test_ensemble_model(ensemble_model, num_games=5):
    env = SnakeGame(window_id=0)  # We'll use a single window for testing
    state_size = 13
    action_size = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model.to(device)

    for game in range(num_games):
        state = env.reset()
        done = False
        score = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = ensemble_model(state_tensor)
            action = torch.argmax(action_values).item()
            next_state, reward, done, game_score = env.play_step(np.eye(action_size)[action])
            state = next_state
            score += reward

            env._update_ui()
            pygame.time.delay(50)  # Adjust delay for visualization speed

        print(f"Game {game + 1}: Score = {game_score}")
    
    pygame.quit()

# Modify the main block to use the test_ensemble_model function
if __name__ == "__main__":
    num_gpus = 4  # Set to 4 for NVIDIA A6000 Ada GPUs
    print(f"Training on {num_gpus} GPUs")

    # Check if a saved model exists
    if os.path.exists('snake_agent_ensemble.pth'):
        print("Loading existing ensemble model for retraining...")
        existing_ensemble = load_ensemble_model('snake_agent_ensemble.pth')
        ensemble_model = train_parallel(num_gpus, existing_ensemble=existing_ensemble)
    else:
        print("No existing model found. Starting training from scratch...")
        ensemble_model = train_parallel(num_gpus)

    # Add debugging prints
    print(f"Ensemble model created with {len(ensemble_model.models)} models")
    for i, model in enumerate(ensemble_model.models):
        print(f"Model {i}: {type(model)}")

    # Test the ensemble model
    print("Testing the trained ensemble model...")
    test_ensemble_model(ensemble_model)
