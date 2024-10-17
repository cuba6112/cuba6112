import os  # Added to check for existing model file
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import pygame
import matplotlib.pyplot as plt

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

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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
        reward = 0
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
        ]

        return np.array(state, dtype=float)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Adjusted for a reasonable decay rate
        self.learning_rate = 0.001  # Adjusted learning rate
        self.model = DQN(state_size, 256, action_size)
        self.target_model = DQN(state_size, 256, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()

def train_dqn():
    env = SnakeGame()
    agent = DQNAgent(11, 3)  # Adjusted state_size to 11
    batch_size = 64
    n_episodes = 1000  # Adjust as needed
    scores = []
    display_every = 100  # Display the game every 100 episodes
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
        render = (e % display_every == 0)
        env.display_ui = render  # Control UI display based on render

        while not done:
            action = agent.act(state)
            next_state, reward, done, score = env.play_step(np.eye(3)[action])

            if render:
                env._update_ui()
                pygame.time.delay(50)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay(batch_size)

        if e % 10 == 0:  # Update target network every 10 episodes
            agent.update_target_model()

        if render:
            pygame.time.delay(500)

        print(f"Episode: {e}/{n_episodes}, Score: {score}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        scores.append(score)

        # Early stopping condition (optional)
        if e > 100 and np.mean(scores[-100:]) > 10:
            print(f"Solved in {e} episodes!")
            break

    # Save the retrained model
    agent.save(model_filename)
    print(f"Model saved as '{model_filename}' after retraining.")

    plt.plot(scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()

    # After training, play one game with the retrained agent
    play_with_trained_model(env)

def play_with_trained_model(env):
    # Load the trained model
    agent = DQNAgent(11, 3)
    agent.load('snake_agent.pth')
    agent.epsilon = 0.0  # Set epsilon to 0 to disable random actions

    state = env.reset()
    done = False
    env.display_ui = True  # Ensure UI is displayed during gameplay

    while not done:
        action = agent.act(state, use_epsilon=False)
        next_state, reward, done, score = env.play_step(np.eye(3)[action])
        state = next_state
        env._update_ui()
        pygame.time.delay(100)  # Slow down the game to make it visible

    print(f"Game Over. Final Score: {score}")
    pygame.time.delay(2000)  # Wait for 2 seconds before closing

if __name__ == "__main__":
 #   train_dqn()


    #Initialize environment
    env = SnakeGame()
    # Play with the trained model
    play_with_trained_model(env)