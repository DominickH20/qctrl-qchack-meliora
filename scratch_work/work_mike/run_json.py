import numpy as np
import matplotlib.pyplot as plt
import sys

import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
import jsonpickle

from qctrlvisualizer import get_qctrl_style, plot_controls
from qctrl import Qctrl

#configure environment
from dotenv import dotenv_values
config = dotenv_values(".env")
import real_q


#open file and decode it
if (len(sys.argv) < 2):
    print("Enter a json pickle path as first arg!")
    exit(0)

path = sys.argv[1]
print(path)
circuit = "H"
if ("not" in path):
    circuit = "NOT"

json_encode = None
with open(path, 'r') as file:
    json_encode = file.read()

json_decode = jsonpickle.decode(json_encode)


#access computer
qctrl = Qctrl(email=config['EMAIL'], password=config['PW'])

#build controls
duration = json_decode["duration"]
values = json_decode["values"]
shot_count = 1024
repetitions = [1, 4, 16, 32, 64]
controls = []

# Iterate through possible repetitions
for rep in repetitions:
    controls.append({"duration": duration, "values": values, "repetition_count": rep})

#run experiment
experiment_results = qctrl.functions.calculate_qchack_measurements(
    controls=controls,
    shot_count=shot_count,
)
measurements = experiment_results.measurements
for k, measurement_counts in enumerate(measurements):
    print(f"control #{k}: {measurement_counts}")
    
for k, measurement_counts in enumerate(measurements):
    p0 = measurement_counts.count(0) / shot_count
    p1 = measurement_counts.count(1) / shot_count
    p2 = measurement_counts.count(2) / shot_count
    print(f"control #{k}: P(|0>) = {p0:.2f}, P(|1>) = {p1:.2f}, P(|2>) = {p2:.2f}")

#conduct experiment
repetitions, experiment_results = repetitions * 1, experiment_results
repetitions = np.split(np.array(repetitions), 1)
measurements = np.split(np.array(experiment_results.measurements), 1)

#build losses constructed off of probabilty outputs
losses = []
loss_list = []
for i in range(len(repetitions)):
    print("Control # {}".format(i + 1))
    loss_sum = 0
    for repetition_count, measurement_counts in zip(
        repetitions [i], measurements [i]
    ):
        measurement_counts = list(measurement_counts)
        p0 = measurement_counts.count(0) / shot_count
        p1 = measurement_counts.count(1) / shot_count
        p2 = measurement_counts.count(2) / shot_count

        print(
            f"With {repetition_count:2d} repetitions: P(|0>) = {p0:.2f}, P(|1>) = {p1:.2f}, P(|2>) = {p2:.2f}"
        )

        #construct and store loss
        if circuit == "H":
            loss_sum += ((p0 - 0.5) ** 2) / repetition_count
            loss_sum += ((p1 - 0.5) ** 2) / repetition_count
            loss_list.append(((p0 - 0.5) ** 2) / repetition_count + ((p1 - 0.5) ** 2) / repetition_count)
        elif circuit == "NOT":
            loss_sum += ((p0 - 0) ** 2) / repetition_count
            loss_sum += ((p1 - 1) ** 2) / repetition_count
            loss_list.append(((p0 - 0) ** 2) / repetition_count + ((p1 - 1) ** 2) / repetition_count)

    #output loss list and the best total loss (sum of all of them)
    losses += [loss_sum]
    print("LOSS LIST: ", loss_list)
    print("BEST LOSS: ", losses[0]) #there is one loss