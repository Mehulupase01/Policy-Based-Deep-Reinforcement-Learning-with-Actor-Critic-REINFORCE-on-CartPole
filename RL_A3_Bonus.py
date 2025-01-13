import cma
import numpy as np

def objective_function(x):
    return np.sum(np.square(x))

x0 = np.zeros(10)
sigma0 = 0.5

es = cma.CMAEvolutionStrategy(x0, sigma0)
while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [objective_function(x) for x in solutions])
    es.logger.add()
    es.disp()

print("Best solution found:", es.result.xbest)



import gym
import numpy as np
import cma

# Initialize the gym environment
env = gym.make('CartPole-v1')

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def policy(parameters, observation):
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    action_probs = policy_network(observation_tensor)
    action = 1 if action_probs.item() > 0.5 else 0
    return action

policy_network = PolicyNetwork(env.observation_space.shape[0], 64, 1)

def objective_function(parameters):
    total_rewards = []
    for _ in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = policy(parameters, observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
    return -np.mean(total_rewards)

import numpy as np
import cma
import gym
import torch
import torch.nn as nn

env = gym.make('CartPole-v1')

num_episodes = 10
max_steps = 500

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def policy(parameters, observation):
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    action_probs = policy_network(observation_tensor)
    action = 1 if action_probs.item() > 0.5 else 0
    return action

def objective_function(parameters):
    total_rewards = []
    for _ in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = policy(parameters, observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
    return -np.mean(total_rewards)

#CMA-ES
x0 = np.zeros(env.observation_space.shape[0])
sigma0 = 0.5
es = cma.CMAEvolutionStrategy(x0, sigma0)

#policy network
policy_network = PolicyNetwork(env.observation_space.shape[0], 64, 1)

while not es.stop():
    solutions = es.ask()
    fitness_values = [objective_function(x) for x in solutions]
    es.tell(solutions, fitness_values)

    es.logger.add()
    es.disp()

print("Best solution found:", es.result.xbest)

"""**Task 1 - CMA-ES**"""


"""**Smoothened**"""

import numpy as np
import cma
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def smooth(data, window_width):
    """ Smoothen data using a moving average filter. """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

env = gym.make('CartPole-v1')

num_episodes = 10
max_steps = 500

# Policy network class
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Policy function
def policy(parameters, observation):
    observation_tensor = torch.tensor(observation, dtype=torch.float32)
    action_probs = policy_network(observation_tensor)
    action = 1 if action_probs.item() > 0.5 else 0
    return action

# Objective function
def objective_function(parameters):
    total_rewards = []
    for _ in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = policy(parameters, observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
    return -np.mean(total_rewards)

x0 = np.zeros(env.observation_space.shape[0])
sigma0 = 0.5
es = cma.CMAEvolutionStrategy(x0, sigma0)

policy_network = PolicyNetwork(env.observation_space.shape[0], 64, 1)

writer = SummaryWriter()

episode_rewards = []
while not es.stop():
    solutions = es.ask()
    fitness_values = [objective_function(x) for x in solutions]
    es.tell(solutions, fitness_values)

    best_reward = max(-np.min(fitness_values), max(episode_rewards, default=0))
    writer.add_scalar('Training/Best_Reward', best_reward, es.countiter)

    episode_rewards.extend([-fitness for fitness in fitness_values])

    print(f"Iteration: {es.countiter}, Best Reward: {best_reward}")

writer.close()

print("Best solution found:", es.result.xbest)

# Smoothing the episode rewards
window_width = 10  # Adjust window width for desired smoothing
smoothed_rewards = smooth(episode_rewards, window_width)

plt.plot(smoothed_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Smoothed Episode Rewards over Optimization Iterations')
plt.show()

"""**Task 2 - Clip-PPO**

**Smoothened**
"""

import matplotlib.pyplot as plt
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class PolicyModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyModel, self).__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(num_actions, activation='softmax')
        self.value = Dense(1)

    def call(self, state):
        x = self.dense1(state)
        probs = self.dense2(x)
        value = self.value(x)
        return probs, value

def smooth(data, window_width):
    """ Smoothen data using a moving average filter. """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

def train(env_name='CartPole-v1', num_episodes=1000, max_steps=1000, gamma=0.99, learning_rate=0.01, clip_ratio=0.2):
    env = gym.make(env_name)
    model = PolicyModel(env.action_space.n)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # For plotting
    total_rewards = []
    policy_losses = []
    value_losses = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards, episode_states, episode_actions, episode_probs, episode_values = [], [], [], [], []

        for step in range(max_steps):
            state = np.expand_dims(state, axis=0).astype(np.float32)
            probs, value = model(state)
            action = np.random.choice(env.action_space.n, p=probs.numpy()[0])

            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            episode_states.append(state)
            episode_actions.append(action)
            episode_probs.append(probs[0, action])
            episode_values.append(value[0, 0])

            state = next_state
            if done:
                break

        discounted_rewards = [np.sum(episode_rewards[i:] * (gamma ** np.arange(len(episode_rewards) - i))) for i in range(len(episode_rewards))]

        # Preparing for loss calculation
        with tf.GradientTape() as tape:
            states_tensor = tf.concat(episode_states, axis=0)
            probs, values = model(states_tensor)

            indices = np.array(episode_actions)
            action_probs = tf.gather(probs, indices, axis=1, batch_dims=1)
            old_action_probs = tf.convert_to_tensor(episode_probs, dtype=tf.float32)

            advantages = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32) - values
            ratio = action_probs / old_action_probs
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(discounted_rewards, dtype=tf.float32) - values))
            total_loss = policy_loss + value_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_rewards.append(np.sum(episode_rewards))
        policy_losses.append(policy_loss.numpy())
        value_losses.append(value_loss.numpy())

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {np.sum(episode_rewards)}, Policy Loss: {policy_loss.numpy()}, Value Loss: {value_loss.numpy()}")

    # Plotting
    window_width = 50  # Width of the moving average window
    smoothed_rewards = smooth(total_rewards, window_width)
    smoothed_policy_losses = smooth(policy_losses, window_width)
    smoothed_value_losses = smooth(value_losses, window_width)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(smoothed_rewards, label='Total Rewards (smoothed)')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(smoothed_policy_losses, label='Policy Loss (smoothed)')
    plt.plot(smoothed_value_losses, label='Value Loss (smoothed)')
    plt.title('Policy and Value Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()