import numpy as np
import matplotlib.pyplot as plt

import genetic_gaussian_search as search
import real_q

def load_seed(filename):
    s = np.load(filename)
    return s

def QCTRL_loss(controls, params):
    repetitions, experiment_results = real_q.run_on_q(controls, params)

    repetitions = np.split(np.array(repetitions), len(controls))
    measurements = np.split(np.array(experiment_results.measurements), len(controls))
    losses = []
    for i in range(len(repetitions)):
        if params["verbose"]: print("Control # {}".format(i + 1))
        loss_sum = 0
        for repetition_count, measurement_counts in zip(
            repetitions [i], measurements [i]
        ):
            measurement_counts = list(measurement_counts)
            p0 = measurement_counts.count(0) / params ["shot_count"]
            p1 = measurement_counts.count(1) / params ["shot_count"]
            p2 = measurement_counts.count(2) / params ["shot_count"]

            if (params["verbose"]): 
                print(
                    f"With {repetition_count:2d} repetitions: P(|0>) = {p0:.2f}, P(|1>) = {p1:.2f}, P(|2>) = {p2:.2f}"
                )

            if params["circuit"] == "H":
                loss_sum += ((p0 - 0.5) ** 2) / repetition_count
                loss_sum += ((p1 - 0.5) ** 2) / repetition_count
            elif params["circuit"] == "NOT":
                loss_sum += ((p0 - 0) ** 2) / repetition_count
                loss_sum += ((p1 - 1) ** 2) / repetition_count
        losses += [loss_sum]
    return losses


#initialize parameters
gate_type = "H" #H or NOT
seed = load_seed(gate_type+"_START_S.npy") #load_seed(gate_type+"_START_U.npy")
segment_count = seed.shape[0]
search_params = {
    "gaussian_params" : {
        "amp_mean": 0,
        "ph_mean": 0,
        "amp_sd": .1,
        "ph_sd": .6,
    },
    "iterations": 3,
    "population size": 10, #must be even!!
    "crossover prob": 0.1,
    "mutation prob amp": 1/(segment_count*2),
    "mutation prob phase": 1/(segment_count*2)
}

max_drive_amplitude = 2 * np.pi * 20                       # MHz
loss_params = {
    "duration": 5 * np.pi / (max_drive_amplitude) * 1000,  # Convert to ns
    "shot_count": 10,
    "verbose": False,
    "circuit": gate_type
}

best, score = search.genetic_gaussian_search(QCTRL_loss, seed, search_params, loss_params)
print('Done! THE BEST IS:')
print('f(%s \n) = %f' % (search.np_2d_print(best), score))
real_q.print_results_single(best, loss_params)