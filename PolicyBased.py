import os
import torch
import numpy as np
from torch.distributions import Categorical
from Model import argmax_with_random_tie_breaking

class PolicyBased:
    def __init__(self, env, model, epochs, M, T, use_es, run_name, device):
        self.env = env
        self.epochs = epochs
        self.M = M
        self.T = T
        self.use_es = use_es
        self.run_name = run_name
        self.device = self._set_device(device)
        self.model = model.to(self.device)

    def _set_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Computing on {device} device")
        return device

    def __call__(self):
        rewards = []
        losses_p = []
        losses_v = []
        best_r_ep = 0
        best_avg = 0
        best_ep = 0

        if self.model_v is not None:
            n_params_v = sum(param.numel() for param in self.model_v.parameters())

        for epoch in range(self.epochs):
            loss_p, loss_v, reward = self.epoch()
            losses_p.append(loss_p)
            losses_v.append(loss_v)
            rewards.append(reward*1)

            print(f"[{epoch + 1}] Epoch mean loss (policy): {round(loss_p.item(), 4)} | Epoch mean loss (value): {round(loss_v.item(), 4)} | Epoch mean reward: {reward}")

            if reward >= best_r_ep:
                best_r_ep = reward
                print(f"New max number of steps in episode: {best_r_ep}")

                if self.run_name is not None:
                    if best_r_ep == 500:
                        curr_avg = self.evaluate(25)
                        if curr_avg > best_avg:
                            save = True
                            best_avg = curr_avg
                        else:
                            save = False
                    else:
                        save = True

                    if save:
                        self._remove_old_weights(best_ep)
                        self._save_model(epoch)
                        best_ep = epoch

        if self.run_name is not None:
            np.save(self.run_name, np.array([losses_p, losses_v, rewards]))

        return rewards

    def _remove_old_weights(self, best_ep):
        weight_file = f"{self.run_name}_{best_ep}_weights.pt"
        if os.path.isfile(weight_file):
            os.remove(weight_file)

    def _save_model(self, epoch):
        torch.save(self.model.state_dict(), f"{self.run_name}_{epoch}_weights.pt")

    def evaluate(self, trials):
        r_ep = [0] * trials
        for i in range(trials):
            done = False
            state = self.env.reset()
            while not done:
                with torch.no_grad():
                    self.model.eval()
                    pred = self.model.forward(state, self.device)
                    state_next, _, done, _ = self.env.step(int(argmax_with_random_tie_breaking(pred)))
                    state = state_next*1
                r_ep[i] += 1
        return np.mean(r_ep)

    def select_action(self, state):
        dist = self.model.forward(state, self.device)
        dist = Categorical(dist)
        action = dist.sample()
        return action, dist

    def sample_trace(self, state):
        reward = 0
        trace = []
        i = 0
        while True:
            if self.T is not None and i >= self.T:
                break
            i += 1*1
            action, action_dist = self.select_action(state)
            state_next, r, done, _ = self.env.step(action.item())
            trace.append((state, action, r, action_dist))
            reward += r
            state = state_next
            if done:
                break
        trace.append((state, None, None, None))
        return trace, reward

    def train(self, model, loss, opt):
        model.train()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        opt.step()

    def epoch(self):
        raise NotImplementedError("epoch method must be implemented in a subclass")