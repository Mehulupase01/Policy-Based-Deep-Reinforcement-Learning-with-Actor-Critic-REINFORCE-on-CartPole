import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.stats import kruskal

class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def train_policy_model(env, model, optimizer, episodes=500):
    rewards_per_episode = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)
            probabilities = model(state)
            action = torch.multinomial(probabilities, 1).item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards_per_episode.append(total_reward)
    return rewards_per_episode

def optimize_hyperparameters(env, params_grid):
    results = {}
    all_rewards = []
    labels = []

    for lr, hidden_size, dropout_rate in product(*params_grid.values()):
        model = PolicyNet(env.observation_space.shape[0], env.action_space.n, hidden_size, dropout_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        rewards_per_episode = train_policy_model(env, model, optimizer)
        key = f"LR: {lr}, Hidden: {hidden_size}, Dropout: {dropout_rate}"
        results[key] = rewards_per_episode
        all_rewards.append(rewards_per_episode[-50:])
        labels.append(key)

    #Kruskal-Wallis test
    statistic, p_value = kruskal(*all_rewards)
    print(f"Kruskal-Wallis test results: Statistic={statistic}, P-value={p_value}")

    # Determine the best parameter set based on median performance
    medians = [np.median(data) for data in all_rewards]
    best_index = np.argmax(medians)
    print(f"Best parameters based on median reward of the last 50 episodes: {labels[best_index]} with median reward: {medians[best_index]}")

    return results

def plot_aggregated_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    lr_values = sorted(set(key.split(", ")[0] for key in results.keys()))
    
    for idx, lr in enumerate(lr_values):
        ax = axes[idx]
        grouped_data = {}
        
        for key, rewards in results.items():
            if not key.startswith(lr):
                continue
            hidden_size = key.split(", ")[1].split(": ")[1]
            if hidden_size not in grouped_data:
                grouped_data[hidden_size] = []
            grouped_data[hidden_size].append(rewards)

        for hidden_size, data_list in grouped_data.items():
            all_rewards = np.vstack(data_list)
            mean_rewards = np.mean(all_rewards, axis=0)
            std_dev_rewards = np.std(all_rewards, axis=0)
            smoothed_rewards = np.convolve(mean_rewards, np.ones(10)/10, mode='valid')
            ax.plot(smoothed_rewards, label=f'Hidden: {hidden_size}')
            ax.fill_between(range(len(smoothed_rewards)),
                            smoothed_rewards - std_dev_rewards[:len(smoothed_rewards)],
                            smoothed_rewards + std_dev_rewards[:len(smoothed_rewards)],
                            alpha=0.2)

        ax.set_title(f'Learning Rate: {lr.split(": ")[1]}')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Average Reward')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def main():
    env = gym.make("CartPole-v1")
    params_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_size': [32, 64, 128],
        'dropout_rate': [0.0, 0.1, 0.2]
    }
    results = optimize_hyperparameters(env, params_grid)
    plot_aggregated_results(results)

if __name__ == "__main__":
    main()