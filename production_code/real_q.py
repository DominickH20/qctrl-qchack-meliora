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
        values = wave [0] * np.exp(1j * wave [1])

        # Iterate through possible repetitions
        for rep in repetitions:
            controls.append({"duration": duration, "values": values, "repetition_count": rep})

    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls=controls,
        shot_count=shot_count,
    )

    return repetitions * len(waves_list), experiment_results
