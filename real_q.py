#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import get_qctrl_style, plot_controls
from qctrl import Qctrl

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

def run_on_q(phases, amplitudes, params):
    qctrl = Qctrl()

    control_count = params ["control_count"]
    segment_count = params ["segment_count"]
    duration = params ["duration"]
    shot_count = params ["shot_count"]

    # We now create a list of controls to send to the qubit. Each of them is a dictionary with a `duration` (how long the pulse is) and an array of (complex) `values` indicating the strength of the pulse in piecewise-constant intervals. Here we use random pulses, so we do not expect them to perform very well at all or implement any particular gate.

    controls = []
    for k in range(control_count):
        # Create a random string of complex numbers for each controls.
        values = amplitudes * np.exp(1j * phases)

        controls.append({"duration": duration, "values": values})

    # Plot the last control as an example.
    plot_controls(
        figure=plt.figure(),
        controls={
            "$\Omega$": [
                {"duration": duration / segment_count, "value": value} for value in values
            ]
        },
    )


    # We can now send those controls to the qubit and get back the results of applying each one of them. We put the returned object in `experiment_results`.

    # Obtain the results of the experiment.
    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls=controls,
        shot_count=shot_count,
    )


    # What we are interested in are the results of the measurements on the qubit, which are in `experiment_results.measurements`. This is a list containing, for each control that we have sent to the qubit, the result of `shot_count` measurements, that is, whether the qubit was found to be in state $|0\rangle$, $|1\rangle$, or $|2\rangle$.

    measurements = experiment_results.measurements

    repetitions = [1, 4, 16, 32, 64]

    controls = []
    # Create a random string of complex numbers for all control,
    # but set a different repetition_count for each control.
    values = amplitudes * np.exp(1j * phases)

    for repetition_count in repetitions:
        controls.append(
            {"duration": duration, "values": values, "repetition_count": repetition_count}
        )

    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls=controls,
        shot_count=shot_count,
    )

    return repetitions, experiment_results
