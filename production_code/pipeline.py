import numpy as np
import matplotlib.pyplot as plt
from qctrlvisualizer import get_qctrl_style, plot_controls
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import jsonpickle

import genetic_gaussian_search as search
import real_q

#function to load array from filename
def load_seed(filename):
    s = np.load(filename)
    return s

#Loss function that queries qubit (for use in genetic algo)
def QCTRL_loss(controls, params):
    #get experiment results
    repetitions, experiment_results = real_q.run_on_q(controls, params)

    #break down results into probabilties and compute loss
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
            #loss is computed here
            if params["circuit"] == "H":
                loss_sum += ((p0 - 0.5) ** 2) / repetition_count
                loss_sum += ((p1 - 0.5) ** 2) / repetition_count
            elif params["circuit"] == "NOT":
                loss_sum += ((p0 - 0) ** 2) / repetition_count
                loss_sum += ((p1 - 1) ** 2) / repetition_count
        #aggregate over all repetitions
        losses += [loss_sum]
    return losses


#initialize parameters for SEARCH
gate_type = "H" #"H" or "NOT"
seed = load_seed(gate_type+"_START_S_BEST.npy") #load_seed(gate_type+"_START_U.npy")
segment_count = seed.shape[0]
search_params = {
    "gaussian_params" : {
        "amp_mean": 0,
        "ph_mean": 0,
        "amp_sd": .1,
        "ph_sd": .6,
    },
    "iterations": 35,
    "population size": 10, #must be even!!
    "crossover prob": 0.5,
    "mutation prob amp": 1/(segment_count*2),
    "mutation prob phase": 1/(segment_count*2)
}

#init parameters for LOSS
max_drive_amplitude = 2 * np.pi * 20                       # MHz
loss_params = {
    "duration": 5 * np.pi / (max_drive_amplitude) * 1000,  # Convert to ns
    "shot_count": 1024,
    "verbose": True,
    "circuit": gate_type
}

#conduct search
best, score = search.genetic_gaussian_search(QCTRL_loss, seed, search_params, loss_params)
print('Done! THE BEST IS:')
print('f(%s \n) = %f' % (search.np_2d_print(best), score))
real_q.print_results_single(best, loss_params)

#output a json, normalize just in case
max_amp = max(best[:,0])
best[:,0] = best[:,0]/max_amp
values = best[:,0] * np.exp(1j * best[:,1])

#build json and output
json_out = {
    "duration" : loss_params["duration"],
    "values": values
}
# print(json_out)
json_encode = jsonpickle.encode(json_out)
with open(gate_type.lower()+"_gate_pulse.json", 'w') as file:
    file.write(json_encode)

#plot our pulse!
plot_controls(
    figure=plt.figure(),
    controls={
        "$\Omega$": [
            {"duration": json_out["duration"] / segment_count / 1e9, "value": value}
            for value in json_out["values"]
        ]
    },
)
plt.show()