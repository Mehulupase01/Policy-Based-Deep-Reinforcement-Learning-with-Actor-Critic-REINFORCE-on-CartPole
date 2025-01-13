# Policy-Based Deep Reinforcement Learning with Actor-Critic REINFORCE on CartPole
 This project implements Policy-Based Reinforcement Learning algorithms like REINFORCE and Actor-Critic with bootstrapping, baseline subtraction, and entropy regularization. The environment used is CartPole-v1. The goal is to investigate the effect of entropy on exploration and compare various configurations of the Actor-Critic algorithm to improve stability and performance

# Policy-Based Deep Reinforcement Learning with Actor-Critic and REINFORCE on CartPole

This project implements **Policy-Based Reinforcement Learning** algorithms, including **REINFORCE** and **Actor-Critic** with various configurations such as **bootstrapping**, **baseline subtraction**, and **entropy regularization** in the **CartPole-v1** environment. These techniques aim to improve the stability and performance of policy gradient methods by addressing the high variance typically associated with them.

## Overview

### Goal:
1. Implement **REINFORCE** and **Actor-Critic** algorithms to solve the **CartPole-v1** environment.
2. Experiment with different configurations of **Actor-Critic**: 
   - Actor-Critic with bootstrapping.
   - Actor-Critic with baseline subtraction.
   - Actor-Critic with both bootstrapping and baseline subtraction.
3. Use **entropy regularization** as an exploration method to balance exploration and exploitation.
4. Investigate the effects of **hyperparameter tuning** (e.g., learning rate, discount factor) on performance.

### Algorithms:
1. **REINFORCE**: A Monte Carlo Policy Gradient method that updates the policy by considering complete episodes and their accumulated rewards.
   
   **REINFORCE Update Rule**:
   \[
   \theta_{t+1} = \theta_t + \alpha \sum_{t=0}^{T} \gamma^t G_t \nabla_{\theta} \log \pi_{\theta}(A_t | S_t)
   \]
   where \( G_t \) is the total discounted reward from time step \( t \).

2. **Actor-Critic**:
   - **With Bootstrapping**: Updates the policy using a value function (critic) to reduce variance in the policy gradient estimate.
   - **With Baseline Subtraction**: Reduces variance further by subtracting the value function (critic) estimate from the reward.
   - **With Both**: Combines bootstrapping and baseline subtraction for optimal performance.

### Hyperparameters:
The experiments are conducted with various hyperparameter configurations. Some key hyperparameters include:
- **Learning Rate (α)**: The rate at which the model parameters are updated.
- **Discount Factor (γ)**: Determines the importance of future rewards.
- **Entropy Factor (η)**: Controls the strength of the entropy regularization to encourage exploration.
- **Epochs**: Number of training episodes.
- **Batch Size**: Number of episodes per gradient update.

| Hyperparameter           | Value Range      |
|--------------------------|------------------|
| **Learning Rate (α)**     | 0.0002 - 0.004   |
| **Discount Factor (γ)**   | 0.5, 0.9         |
| **Entropy Factor (η)**    | 0.2 - 0.9        |
| **Batch Size (M)**        | 32, 64, 128      |
| **Bootstrapping Depth (n)** | 50 to 250      |

### Code Structure:
This project is organized into several Python files, each responsible for different parts of the implementation:

- **`AC_bootstrap.py`**: Implements the **Actor-Critic** algorithm with bootstrapping and baseline subtraction.
- **`Reinforce.py`**: Implements the **REINFORCE** algorithm.
- **`cartpole_feature_ablation.py`**: Contains experiments related to **feature ablation** for the CartPole task.
- **`cartpole_feature_sensitivity.py`**: Contains experiments related to **feature sensitivity** in CartPole.
- **`cartpole_hyperparameter_optimization.py`**: Hyperparameter optimization experiments for **CartPole**.
- **`cartpole_parameter_impact.py`**: Examines the impact of different parameters on **CartPole** performance.
- **`experiments.py`**: Contains the code to run experiments, tune hyperparameters, and log the results.
- **`Model.py`**: Contains the model architecture for the **policy network** and **value network**.
- **`PolicyBased.py`**: Implements the **Policy-Based** reinforcement learning algorithm.
- **`re_plot.py`**: Used to plot the results of experiments and visualize learning curves.


### Experimentation:
1. **Hyperparameter Tuning**: We systematically adjust learning rates, discount factors, and network architectures to determine the optimal configuration for the CartPole environment.
2. **Performance Metrics**: The performance of each algorithm is evaluated using the average reward per episode, stability (variance), and convergence speed.

### Results & Discussion:
- **REINFORCE**: High variance and slow convergence due to the Monte Carlo method of estimating gradients.
- **Actor-Critic**: Improved stability with bootstrapping and baseline subtraction. These configurations significantly reduce the variance of the policy gradient estimates.
- **Entropy Regularization**: Helps prevent premature convergence by encouraging exploration, especially in early training phases.
- **Best Performing Configuration**: The combination of **bootstrapping**, **baseline subtraction**, and **entropy regularization** resulted in the best overall performance, demonstrating faster convergence and more stable learning.

### Performance Visualization:
The experiment results are plotted using **Matplotlib** to compare the learning curves, policy loss, and value loss over training episodes. Performance plots show the **total reward per episode** and the **loss curves** for both the policy and the value network.

### How to Run:
1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. Run Experiments: To run all the experiments and reproduce the results:

   ```bash
   bash all_experiments.sh

   ```
3. Plot Results: The re_plot.py file can be used to plot the learning curves and other performance metrics for each experiment.

## Conclusion:
This project demonstrates the application of **Policy-Based Reinforcement Learning** algorithms, with a particular focus on the **Actor-Critic** method. The use of **entropy regularization**, **bootstrapping**, and **baseline subtraction** significantly improved the performance and stability of the agent in the **CartPole** environment. The combination of these techniques allowed for more stable learning and faster convergence, with **annealing ε-greedy** providing the best exploration-exploitation balance. The **Clip-PPO** and **CMA-ES** bonus tasks further improved policy updates, adding stability and robustness to the model.

## References:
1. **OpenAI Gym**: [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/)
2. **Deep Q-Learning (Mnih et al., 2015)**: [https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602)
3. **Reinforcement Learning: An Introduction (Sutton & Barto, 2018)**: [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
4. **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: [https://arxiv.org/abs/1604.00702](https://arxiv.org/abs/1604.00702)
5. **Proximal Policy Optimization (PPO)**: [OpenAI Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
