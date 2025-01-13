import argparse
import time
import torch
import gym
from utils import *
from Model import *
from Reinforce import Reinforce
from AC_bootstrap import ACBootstrap

def get_args():
    arg_parser = argparse.ArgumentParser()

    # Model parameters
    arg_parser.add_argument('-exp_name', type=str, default=None)
    arg_parser.add_argument('-optimizer', type=str, default='adam')
    arg_parser.add_argument('-optimizer_lr', type=float, default=0.0003)
    arg_parser.add_argument('-optimizer_v', type=str, default='adam')
    arg_parser.add_argument('-optimizer_lr_v', type=float, default=0.0003)
    arg_parser.add_argument('-computing_device', type=str, default="cuda")

    # Algorithm parameters
    arg_parser.add_argument('-algorithm', type=str, default='reinforce')
    arg_parser.add_argument('-traces', type=int, default=5)
    arg_parser.add_argument('-traces_len', type=int, default=500)
    arg_parser.add_argument('-epochs', type=int, default=1000)
    arg_parser.add_argument('-depth', type=int, default=10)
    arg_parser.add_argument('-gamma', type=float, default=0.5)
    arg_parser.add_argument('-baseline', action='store_true')
    arg_parser.add_argument('-entropy', action='store_true')
    arg_parser.add_argument('-entropy_factor', type=float, default=0.2)

    return arg_parser.parse_args()

def main():
    args = get_args()

    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
        'rms': torch.optim.RMSprop
    }

    env = gym.make("CartPole-v1")
    run_name = (args.run_name if args.run_name else "default_run")
    optimum = 500
    n_repetitions = 2

    plot = LearningCurvePlot(title=args.alg.upper())
    plot.add_hline(optimum, label="optimum")

    curve = None
    for i in range(n_repetitions):
        mlp_policy = MLP(4, 2, False)
        opt_policy = optimizers[args.optimizer](mlp_policy.parameters(), args.optim_lr)

        if args.baseline or args.alg == "AC_bootstrap":
            mlp_value = MLP(4, 2, True)
            opt_value = optimizers[args.optimizer_v](mlp_value.parameters(), args.optim_lr_v)
        else:
            mlp_value = None
            opt_value = None

        if args.alg == "reinforce":
            algorithm = Reinforce(env, mlp_policy, opt_policy, mlp_value, opt_value, args.epochs, args.traces,
                                  args.gamma, args.entropy, args.entropy_factor, False, run_name + "_" + str(i), args.device)
        elif args.alg == "AC_bootstrap":
            algorithm = ACBootstrap(env, mlp_policy, opt_policy, mlp_value, opt_value, args.epochs, args.traces, args.trace_len,
                                    args.n, args.baseline, args.entropy, args.entropy_factor, False, run_name + "_" + str(i), args.device)
        else:
            print("Please select a valid model")
            continue

        start_time = time.time()
        rewards = np.array(algorithm())
        print(f"Running one setting takes {round((time.time() - start_time) / 60, 2)} minutes")

        rewards = rewards[:500]

        if curve is None:
            curve = rewards
        else:
            curve += rewards*1

    curve /= n_repetitions
    curve = smooth(curve, 35, 1)
    plot.add_curve(curve, label=r"label")

    plot.save(args.run_name + ".png")
    env.close()

if __name__ == "__main__":
    main()