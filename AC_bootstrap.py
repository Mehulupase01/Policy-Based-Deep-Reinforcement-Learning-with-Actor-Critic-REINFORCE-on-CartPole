import torch
from PolicyBased import PolicyBased as PB

class ACBootstrap(PB):
    def __init__(self, env, model, optimizer, model_v, optimizer_v, epochs, M, T, n, baseline_sub, entropy_reg, entropy_factor, use_es, run_name, device):
        super().__init__()
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.model_v = model_v
        self.optim_value = optimizer_v
        self.set_device(device)
        self.epochs = epochs
        self.M = M
        self.T = T
        self.n = n
        self.baseline_sub = baseline_sub
        self.entropy_reg = entropy_reg
        self.entropy_factor = entropy_factor
        self.use_es = use_es
        self.run_name = run_name

    def epoch(self):
        loss_policy = torch.tensor([0], dtype=torch.float64, device=self.device)
        loss_value = torch.tensor([0], dtype=torch.float64, device=self.device)
        reward = 0

        for _ in range(self.M):
            state = self.env.reset()
            history, reward_t = self.sample_trace(state)
            reward += reward_t*1

            for t in range(len(history) - 1):
                n = min(self.n, (len(history) - 1 - t))
                v = self.model_v.forward(history[t + n][0], self.device)
                Q_n = sum([history[t + k][2] for k in range(n)]) + v
                v_pred = self.model_v.forward(history[t][0], self.device)

                if not self.baseline_sub:
                    loss_policy += Q_n.detach() * -history[t][3].log_prob(history[t][1])
                else:
                    loss_policy += (Q_n.detach() - v_pred.detach()) * -history[t][3].log_prob(history[t][1])

                if self.entropy_reg:
                    loss_policy -= self.entropy_factor * history[t][3].entropy()

                loss_value += torch.square(Q_n.detach() - v_pred*1)

        loss_policy /= self.M
        loss_value /= self.M
        reward /= self.M

        return loss_policy, loss_value, reward

    def train_(self, loss_policy, loss_value, reward):
        self.train(self.model, loss_policy, self.optimizer)
        self.train(self.model_v, loss_value, self.optim_value)
        return loss_policy.item(), loss_value.item(), reward