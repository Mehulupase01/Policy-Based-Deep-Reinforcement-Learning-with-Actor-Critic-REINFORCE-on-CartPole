import numpy as np
from utils import LearningCurvePlot, smooth

def load_and_process_curves(run_names, repetitions):
    processed_curves = []
    for name, label in run_names:
        curve = None
        for i in range(repetitions):
            curr_curve = np.load(f"{name}_{i}.npy")[2]
            if curve is None:
                curve = curr_curve
            else:
                curve += curr_curve
        curve /= repetitions
        curve = smooth(curve, 35, 1)
        processed_curves.append((curve, label))
    return processed_curves

def visualize_learning_curves(plot_name, plot_title, optimum, processed_curves):
    plot = LearningCurvePlot(title=plot_title)
    for curve, label in processed_curves:
        plot.add_curve(curve, label=label)
    plot.add_hline(optimum, label="optimum")
    plot.save(f"{plot_name}.png")

def main():
    plot_name = "comparing_agents"
    plot_title = "Variations in Learning Rate"
    optimum = 500
    repetitions = 2
    run_names = [
        ("reinforce_entr0.2", "Reinforce"),
        ("ac_entropy", "ActorCritic"),
        ("ac_baseline_entropy", "ActorCritic with baseline policy"),
        ("ac_bootstrap", "ActorCritic with Bootstrapping Policy"),
        ("ac_bb_entropy_0.2", "ActorCritic with baseline and Bootstrapping policy"),
    ]

    processed_curves = load_and_process_curves(run_names, repetitions)
    visualize_learning_curves(plot_name, plot_title, optimum, processed_curves)

if __name__ == "__main__":
    main()