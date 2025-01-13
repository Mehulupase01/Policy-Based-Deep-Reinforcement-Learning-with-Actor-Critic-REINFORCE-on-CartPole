import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

class CustomCartPoleEnv(gym.Env):
    def __init__(self, pole_length=None, force_mag=None):
        super(CustomCartPoleEnv, self).__init__()
        self.env = gym.make('CartPole-v1')
        if pole_length is not None:
            self.env.unwrapped.length = pole_length 
        if force_mag is not None:
            self.env.unwrapped.force_mag = force_mag  

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

def train_model(env, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample() 
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return total_rewards 

def main():
    pole_lengths = [0.5, 1.0, 1.5, 2.0]
    force_magnitudes = [10, 20, 30, 40]
    results = {}
    all_rewards = []

    for length in pole_lengths:
        for force in force_magnitudes:
            env = CustomCartPoleEnv(pole_length=length, force_mag=force)
            rewards = train_model(env, episodes=100)
            results[(length, force)] = np.mean(rewards)
            all_rewards.append(rewards) 
            print(f"Tested with pole length {length} and force magnitude {force}: Average Reward = {results[(length, force)]}")

    statistic, p_value = f_oneway(*all_rewards)
    print(f"ANOVA results: Statistic = {statistic}, P-value = {p_value}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for (key, val), color in zip(results.items(), colors):
        ax.bar(f"{key[0]}-{key[1]}", val, color=color, label=f"Length {key[0]}, Force {key[1]}")

    ax.set_xlabel('Configurations (Pole Length, Force Magnitude)')
    ax.set_ylabel('Average Reward')
    ax.set_title('Performance Across Different Dynamics')
    plt.xticks(rotation=45)
    plt.legend(title="Configurations", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("cartpole_dynamics_results.png")
    plt.show()

if __name__ == "__main__":
    main()