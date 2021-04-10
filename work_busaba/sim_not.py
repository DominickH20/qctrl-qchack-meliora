#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import get_qctrl_style, plot_controls
from qctrl import Qctrl

qctrl = Qctrl()

# In[2]:

error_norm =  lambda A, B : 1 - np.abs(np.trace((A.conj().T @ B)) / 2) ** 2

def estimate_probability_of_one(measurements):
    size = len(measurements)
    probability = np.mean(measurements)
    standart_error = np.std(measurements) / np.sqrt(size)
    return (probability, standart_error)

def simulate_more_realistic_qubit(
    duration=1, values=np.array([np.pi]), shots=1024, repetitions=1
):

    # 1. Limits for drive amplitudes
    assert np.amax(values) <= 1.0
    assert np.amin(values) >= -1.0
    max_drive_amplitude = 2 * np.pi * 20  # MHz

    # 2. Dephasing error
    dephasing_error = -2 * 2 * np.pi  # MHz

    # 3. Amplitude error
    amplitude_i_error = 0.98
    amplitude_q_error = 1.03

    # 4. Control line bandwidth limit
    cut_off_frequency = 2 * np.pi * 10  # MHz
    resample_segment_count = 1000

    # 5. SPAM error confusion matrix
    confusion_matrix = np.array([[0.99, 0.01], [0.02, 0.98]])

    # Lowering operator
    b = np.array([[0, 1], [0, 0]])
    # Number operator
    n = np.diag([0, 1])
    # Initial state
    initial_state = np.array([[1], [0]])

    with qctrl.create_graph() as graph:
        # Apply 1. max Rabi rate.
        values = values * max_drive_amplitude

        # Apply 3. amplitude errors.
        values_i = np.real(values) * amplitude_i_error
        values_q = np.imag(values) * amplitude_q_error
        values = values_i + 1j * values_q

        # Apply 4. bandwidth limits
        drive_unfiltered = qctrl.operations.pwc_signal(duration=duration, values=values)
        drive_filtered = qctrl.operations.convolve_pwc(
            pwc=drive_unfiltered,
            kernel_integral=qctrl.operations.sinc_integral_function(cut_off_frequency),
        )
        drive = qctrl.operations.discretize_stf(
            drive_filtered, duration=duration, segments_count=resample_segment_count
        )

        # Construct microwave drive
        drive_term = qctrl.operations.pwc_operator_hermitian_part(
            qctrl.operations.pwc_operator(signal=drive, operator=b)
        )

        # Construct 2. dephasing term.
        dephasing_term = qctrl.operations.constant_pwc_operator(
            operator=dephasing_error * n,
            duration=duration,
        )

        # Construct Hamiltonian.
        hamiltonian = qctrl.operations.pwc_sum(
            [
                drive_term,
                dephasing_term,
            ]
        )

        # Solve Schrodinger's equation and get total unitary at the end
        unitary = qctrl.operations.time_evolution_operators_pwc(
            hamiltonian=hamiltonian,
            sample_times=np.array([duration]),
        )[-1]
        unitary.name = "unitary"

        # Repeat final unitary
        repeated_unitary = np.eye(2)
        for _ in range(repetitions):
            repeated_unitary = repeated_unitary @ unitary
        repeated_unitary.name = "repeated_unitary"

        # Calculate final state.
        state = repeated_unitary @ initial_state

        # Calculate final populations.
        populations = qctrl.operations.abs(state[:, 0]) ** 2
        # Normalize populations
        norm = qctrl.operations.sum(populations)
        populations = populations / norm
        populations.name = "populations"

    # Evaluate graph.
    result = qctrl.functions.calculate_graph(
        graph=graph,
        output_node_names=["unitary", "repeated_unitary", "populations"],
    )

    # Extract outputs.
    unitary = result.output["unitary"]["value"]
    repeated_unitary = result.output["repeated_unitary"]["value"]
    populations = result.output["populations"]["value"]

    # Sample projective measurements.
    true_measurements = np.random.choice(2, size=shots, p=populations)
    measurements = np.array(
        [np.random.choice(2, p=confusion_matrix[m]) for m in true_measurements]
    )

    results = {"unitary": unitary, "measurements": measurements}

    return results

# In[3]:

# 1. Limits for drive amplitudes
max_drive_amplitude = 2 * np.pi * 20  # MHz

# 2. Dephasing error
dephasing_error = -2 * 2 * np.pi  # MHz

# 3. Amplitude error
amplitude_i_error = 0.98
amplitude_q_error = 1.03

# 4. Control line bandwidth limit
cut_off_frequency = 2 * np.pi * 10  # MHz
resample_segment_count = 1000

# Lowering operator
b = np.array([[0, 1], [0, 0]])
# Number operator
n = np.diag([0, 1])
# Initial state
initial_state = np.array([[1], [0]])

# Extra constants used for optimization
# control_count = 5
segment_count = 64 # 16
duration = 5 * np.pi / (max_drive_amplitude) # 30.0
ideal_not_gate = np.array([[0, -1j], [-1j, 0]])


# We now create a list of controls to send to the qubit. Each of them is a dictionary with a `duration` (how long the pulse is) and an array of (complex) `values` indicating the strength of the pulse in piecewise-constant intervals. Here we use random pulses, so we do not expect them to perform very well at all or implement any particular gate.

# In[4]:

with qctrl.create_graph() as graph:
    # Create optimizable modulus and phase.
    values = qctrl.operations.bounded_optimization_variable(
        count=segment_count, lower_bound=0, upper_bound=1,
    ) * qctrl.operations.exp(1j * qctrl.operations.unbounded_optimization_variable(
        count=segment_count, initial_lower_bound=0, initial_upper_bound=2*np.pi,
    ))

    export_drive = qctrl.operations.pwc_signal(
        duration=duration, values=values, name="Omega"
    )

    # Apply 1. max Rabi rate.
    values = values * max_drive_amplitude

    # Apply 3. amplitude errors.
    values_i = np.real(values) * amplitude_i_error
    values_q = np.imag(values) * amplitude_q_error
    values = values_i + 1j * values_q

    # Apply 4. bandwidth limits
    drive_unfiltered = qctrl.operations.pwc_signal(duration=duration, values=values)
    drive_filtered = qctrl.operations.convolve_pwc(
        pwc=drive_unfiltered,
        kernel_integral=qctrl.operations.sinc_integral_function(cut_off_frequency),
    )
    drive = qctrl.operations.discretize_stf(
        drive_filtered, duration=duration, segments_count=resample_segment_count
    )

    # Construct microwave drive
    drive_term = qctrl.operations.pwc_operator_hermitian_part(
        qctrl.operations.pwc_operator(signal=drive, operator=b)
    )

    # Construct 2. dephasing term.
    dephasing_term = qctrl.operations.constant_pwc_operator(
        operator=dephasing_error * n,
        duration=duration,
    )

    # Construct Hamiltonian.
    hamiltonian = qctrl.operations.pwc_sum(
        [
            drive_term,
            dephasing_term,
        ]
    )

    # Construct Target operator
    target_operator = qctrl.operations.target(operator=ideal_not_gate)

    # Calculate infidelity between target NOT gate and final unitary
    indfidelity = qctrl.operations.infidelity_pwc(
        hamiltonian=hamiltonian,
        target_operator=target_operator,
        name="infidelity",
    )

# In[5]:

# Optimize graph
optimization_result = qctrl.functions.calculate_optimization(
    cost_node_name="infidelity",
    output_node_names=["Omega"],
    graph=graph,
)

# In[6]:

print("Infidelity of gate: " + str(optimization_result.cost))

fig = plt.figure()
plot_controls(
    fig,
    controls={
        "$\\Omega$": optimization_result.output["Omega"],
    }, polar=False)
plt.show()

# In[7]:

# Test optimized pulse on more realistic qubit simulation

optimized_values = np.array([segment["value"] for segment in optimization_result.output["Omega"]])
print("Optimized Values:")
print(optimized_values)
result = simulate_more_realistic_qubit(duration=duration, values=optimized_values, shots=1024, repetitions=1)

# In[8]:
realized_not_gate = result["unitary"]
not_error = error_norm(realized_not_gate, ideal_not_gate)

not_measurements = result["measurements"]
not_probability, not_standard_error = estimate_probability_of_one(not_measurements)

print("Realised NOT Gate:")
print(realized_not_gate)
print("Ideal NOT Gate:")
print(ideal_not_gate)
print("NOT Gate Error: " + str(not_error))

# In[9]:

# Normalizing the amplitudes
absolutes = []
for val in optimized_values:
    absolutes += [np.absolute(val)]
max_amp = max(absolutes)

# Write parameters to file
with open("amplitude.txt", "w") as amplitude_f:
    for val in absolutes:
        amplitude_f.write("{}\n".format(val / max_amp))
with open("phase.txt", "w") as phase_f:
    for val in optimized_values:
        phase_f.write("{}\n".format(np.angle(val)))
