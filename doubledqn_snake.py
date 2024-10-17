import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# Snake Game Environment
class SnakeGame:
    def __init__(self):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()
        self.display_ui = False  # Control whether to display the game UI

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
    def __init__(self, state_size, action_size):
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
        self.model = DuelingDQN(state_size, 512, action_size).to(device)
        self.target_model = DuelingDQN(state_size, 512, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss(reduction='none')  # For PER
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.target_model = nn.DataParallel(self.target_model)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Compute TD error for priority
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        action_tensor = torch.tensor([[action]], dtype=torch.long).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)
        done_tensor = torch.tensor([done], dtype=torch.float32).to(device)

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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory.memory) < self.batch_size:
            return

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        minibatch, indices, weights = self.memory.sample(self.batch_size, self.beta)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

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

def train_dqn():
    env = SnakeGame()
    state_size = 13  # Adjusted for additional features
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    n_episodes = 5000  # Increased number of episodes
    scores = []
    display_every = 500  # Display the game every 500 episodes
    model_filename = 'snake_agent.pth'

    # Load existing model if it exists
    if os.path.isfile(model_filename):
        agent.load(model_filename)
        print(f"Loaded existing model '{model_filename}' for retraining.")
        # Optionally adjust epsilon to allow more exploration
        agent.epsilon = max(agent.epsilon, 0.1)
    else:
        print("No existing model found. Starting training from scratch.")

    for e in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # Set to True if you want to display this episode
        render = (e % display_every == 0 and e != 0)
        env.display_ui = render  # Control UI display based on render

        while not done:
            action = agent.act(state)
            next_state, reward, done, score = env.play_step(np.eye(action_size)[action])

            if render:
                env._update_ui()
                pygame.time.delay(50)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

        if e % 10 == 0:  # Update target network every 10 episodes
            agent.update_target_model()

        if render:
            pygame.time.delay(500)

        print(f"Episode: {e}/{n_episodes}, Score: {score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        scores.append(score)

        # Early stopping condition (optional)
        if e > 100 and np.mean(scores[-100:]) > 50:
            print(f"Solved in {e} episodes!")
            break

    # Save the retrained model
    agent.save(model_filename)
    print(f"Model saved as '{model_filename}' after training.")

    plt.plot(scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()

    # After training, play one game with the trained agent
    play_with_trained_model(env)

def play_with_trained_model(env):
    # Load the trained model
    state_size = 13
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.load('snake_agent.pth')
    agent.epsilon = 0.0  # Set epsilon to 0 to disable random actions

    state = env.reset()
    done = False
    env.display_ui = True  # Ensure UI is displayed during gameplay

    while not done:
        action = agent.act(state, use_epsilon=False)
        next_state, reward, done, score = env.play_step(np.eye(agent.action_size)[action])
        state = next_state
        env._update_ui()
        pygame.time.delay(50)  # Slow down the game to make it visible

    print(f"Game Over. Final Score: {score}")
    pygame.time.delay(2000)  # Wait for 2 seconds before closing

if __name__ == "__main__":
    train_dqn()
