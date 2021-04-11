#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import get_qctrl_style, plot_controls
from qctrl import Qctrl

from dotenv import dotenv_values
config = dotenv_values(".env")

# ## Accessing the qubit in the cloud (the challenge!)
# 
# The challenge is to create pulses that implement high fidelity Hadamard and NOT gates on a qubit in the cloud provided through BOULDER OPAL. Here we show you how to send pulses to the qubit in the cloud and get measurement results. 
# 
# In what follows we will show you how to create a series of controls (pulses), send them to the qubit in the cloud, and analyze the results of the experiment. We begin by importing some basic modules and starting a session with the BOULDER OPAL Q-CTRL API.
# 
# We begin by establishing some parameters for our pulses:
# - `control_count`: How many controls we create.
# - `segment_count`: How many segments each pulse is split into.
# - `duration`: The duration (in ns) of each pulse.
# - `shot_count`: How many projective measurements we take at the end of each control.
# 
# You will probably want to change these values later depending on how you approach the challenge. For instance, the larger the `shot_count` is, the more measurements we get out of the qubit for each control (we only keep it small here so that the experiment results are easier to visualize).
#

def run_on_q(waves_list, params):
    qctrl = Qctrl(email=config['EMAIL'], password=config['PW'])

    # Extract parameters
    duration = params ["duration"]
    shot_count = params ["shot_count"]

    repetitions = [1, 4, 16, 32, 64]

    controls = []
    for wave in waves_list:
        # Create a random string of complex numbers for each controls.
        max_amp = max(wave[:,0])
        wave[:,0] = wave[:,0]/max_amp
        values = wave[:,0] * np.exp(1j * wave[:,1])

        # Iterate through possible repetitions
        for rep in repetitions:
            controls.append({"duration": duration, "values": values, "repetition_count": rep})

    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls=controls,
        shot_count=shot_count,
    )

    return repetitions * len(waves_list), experiment_results


def run_on_q_single(wave, params):
    qctrl = Qctrl(email=config['EMAIL'], password=config['PW'])

    # Extract parameters
    duration = params ["duration"]
    shot_count = params ["shot_count"]

    repetitions = [1, 4, 16, 32, 64]

    controls = []

    max_amp = max(wave[:,0])
    wave[:,0] = wave[:,0]/max_amp
    values = wave[:,0] * np.exp(1j * wave[:,1])

    # Iterate through possible repetitions
    for rep in repetitions:
        controls.append({"duration": duration, "values": values, "repetition_count": rep})

    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls=controls,
        shot_count=shot_count,
    )

    return repetitions * 1, experiment_results

def print_results_single(wave, params):
    repetitions, experiment_results = run_on_q_single(wave, params)
    repetitions = np.split(np.array(repetitions), 1)
    measurements = np.split(np.array(experiment_results.measurements), 1)
    losses = []
    loss_list = []
    for i in range(len(repetitions)):
        print("Control # {}".format(i + 1))
        loss_sum = 0
        for repetition_count, measurement_counts in zip(
            repetitions [i], measurements [i]
        ):
            measurement_counts = list(measurement_counts)
            p0 = measurement_counts.count(0) / params ["shot_count"]
            p1 = measurement_counts.count(1) / params ["shot_count"]
            p2 = measurement_counts.count(2) / params ["shot_count"]

            print(
                f"With {repetition_count:2d} repetitions: P(|0>) = {p0:.2f}, P(|1>) = {p1:.2f}, P(|2>) = {p2:.2f}"
            )

            if params["circuit"] == "H":
                loss_sum += ((p0 - 0.5) ** 2) / repetition_count
                loss_sum += ((p1 - 0.5) ** 2) / repetition_count
                loss_list.append(((p0 - 0.5) ** 2) / repetition_count + ((p1 - 0.5) ** 2) / repetition_count)
            elif params["circuit"] == "NOT":
                loss_sum += ((p0 - 0) ** 2) / repetition_count
                loss_sum += ((p1 - 1) ** 2) / repetition_count
                loss_list.append(((p0 - 0) ** 2) / repetition_count + ((p1 - 1) ** 2) / repetition_count)

        losses += [loss_sum]
        print("LOSS LIST: ", loss_list)
        print("BEST LOSS: ", losses[0]) #there is one loss