import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class ModifiedObservationSpace(gym.ObservationWrapper):
    def __init__(self, env, active_features):
        super(ModifiedObservationSpace, self).__init__(env)
        self.active_features = active_features
        self.observation_space = gym.spaces.Box(low=self.observation_space.low[active_features],
                                                high=self.observation_space.high[active_features],
                                                dtype=np.float32)

    def observation(self, observation):
        return observation[self.active_features]

class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def train_policy_model(env, episodes=500):
    model = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    total_rewards = []
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
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)

def main():
    env = gym.make("CartPole-v1")
    features = np.arange(env.observation_space.shape[0])
    feature_names = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity at Tip']
    results = {}

    for i in range(len(features)):
        active_features = features[np.arange(len(features)) != i]
        modified_env = ModifiedObservationSpace(env, active_features)
        print(f"Testing with feature {i} removed.")
        average_reward, std_dev = train_policy_model(modified_env)
        results[feature_names[i]] = (average_reward, std_dev)

    plt.figure(figsize=(12, 6))
    rewards, std_devs = zip(*results.values())
    plt.bar(results.keys(), rewards, yerr=std_devs, capsize=5)
    plt.xlabel('Excluded Feature')
    plt.ylabel('Average Reward')
    plt.title('Feature Ablation Study Results with Learning Policy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("feature_ablation_learning_results.png")
    plt.show()

if __name__ == "__main__":
    main()