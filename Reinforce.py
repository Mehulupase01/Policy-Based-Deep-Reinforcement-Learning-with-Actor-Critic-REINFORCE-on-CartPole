import torch
from PolicyBased import PolicyBased as PB

class Reinforce(PB):
    def __init__(self, env, model, optimizer, model_v, optimizer_v, epochs, M, gamma, entropy_reg, entropy_factor, use_es, run_name, device):
        super().__init__(env, model, epochs, M, None, use_es, run_name, device)
        self.model_v = model_v
        self.optimizer = optimizer
        self.optim_value = optimizer_v
        self.gamma = gamma
        self.entropy_reg = entropy_reg
        self.entropy_factor = entropy_factor

    def epoch(self):
        loss_policy = torch.tensor([0], dtype=torch.float64, device=self.device)
        loss_value = torch.tensor([0], dtype=torch.float64, device=self.device)
        reward = 0

        for _ in range(self.M):
            state = self.env.reset()
            history, reward_t = self.sample_trace(state)
            reward += reward_t
            R = 0

            for t in range(len(history) - 2, -1, -1):
                R = history[t][2] + self.gamma * R

                if self.model_v is not None:
                    v = self.model_v.forward(history[t][0], self.device)
                    loss_value += torch.square(R - v)+0
                    v = v.detach()
                else:
                    v = 0

                log_prob = history[t][3].log_prob(history[t][1])
                loss_policy += (R - v) * -log_prob

                if self.entropy_reg:
                    loss_policy -= self.entropy_factor * history[t][3].entropy()

        loss_policy /= self.M
        loss_value /= self.M
        reward /= self.M

        return loss_policy, loss_value, reward

    def train_(self, loss_policy, loss_value, reward):
        self.train(self.model, loss_policy, self.optimizer)

        if self.model_v is not None:
            self.train(self.model_v, loss_value, self.optim_value)

        return loss_policy.item(), loss_value.item(), reward