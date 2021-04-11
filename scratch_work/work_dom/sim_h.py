#!/usr/bin/env python
# coding: utf-8

import scipy as sc
import numpy as np

def resample (x, k):
  """
  Resample the signal to the given ratio using a sinc kernel

  input:

    x   a vector or matrix with a signal in each row
    k   ratio to resample to

    returns

    y   the up or downsampled signal

    when downsampling the signal will be decimated using scipy.signal.decimate
  """

  if k < 1:
    raise NotImplementedError ('downsampling is not implemented')

  if k == 1:
    return x # nothing to do

  return upsample (x, k)

def upsample (x, k):
  """
  Upsample the signal to the given ratio using a sinc kernel

  input:

    x   a vector or matrix with a signal in each row
    k   ratio to resample to

    returns

    y   the up or downsampled signal

    when downsampling the signal will be decimated using scipy.signal.decimate
  """

  assert k >= 1, 'k must be equal or greater than 1'

  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = n * k

  xt = np.linspace (1, n, n)
  xp = np.linspace (1, n, nn)

  return interp (xp, xt, x)

def interp (xp, xt, x):
  """
  Interpolate the signal to the new points using a sinc kernel

  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on

  output:
  y     the interpolated signal at points xp
  """

  mn = x.shape
  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = len(xp)

  y = np.zeros((m, nn))

  for (pi, p) in enumerate (xp):
    si = np.tile(np.sinc (xt - p), (m, 1))
    y[:, pi] = np.sum(si * x)

  return y.squeeze ()

# In[1]:

# import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import get_qctrl_style, plot_controls
from qctrl import Qctrl

from dotenv import dotenv_values
config = dotenv_values(".env")
qctrl = Qctrl(email=config['EMAIL'], password=config['PW'])

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

def run_main_h ():
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
    segment_count = 32
    duration = 5 * np.pi / (max_drive_amplitude)
    ideal_h_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])


    # We now create a list of controls to send to the qubit. Each of them is a dictionary with a `duration` (how long the pulse is) and an array of (complex) `values` indicating the strength of the pulse in piecewise-constant intervals. Here we use random pulses, so we do not expect them to perform very well at all or implement any particular gate.

    # In[4]:

    with qctrl.create_graph() as graph:
        # Create optimizable modulus and phase.
        values = qctrl.operations.anchored_difference_bounded_variables(
            count=segment_count, lower_bound=0, upper_bound=1,difference_bound = 0.5,
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
        target_operator = qctrl.operations.target(operator=ideal_h_gate)

        # Calculate infidelity between target H gate and final unitary
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
        optimization_count=40,
    )

    # In[6]:

    print("Infidelity of gate: " + str(optimization_result.cost))

    # In[7]:

    # Test optimized pulse on more realistic qubit simulation

    optimized_values = np.array([segment["value"] for segment in optimization_result.output["Omega"]])
    result = simulate_more_realistic_qubit(duration=duration, values=optimized_values, shots=1024, repetitions=1)

    # In[8]:
    realized_h_gate = result["unitary"]
    h_error = error_norm(realized_h_gate, ideal_h_gate)

    h_measurements = result["measurements"]
    h_probability, h_standard_error = estimate_probability_of_one(h_measurements)

    print("Realised H Gate:")
    print(realized_h_gate)
    print("Ideal H Gate:")
    print(ideal_h_gate)
    print("H Gate Error: " + str(h_error))

    # In[81]

    # Perform a Sinc Interpolation on the amplitudes of the pulse

    smoothed_amp = upsample(np.absolute(optimized_values),4)

    smoothed_phase = []
    for i in optimized_values:
        smoothed_phase += [i]
        smoothed_phase += [i]
        smoothed_phase += [i]
        smoothed_phase += [i]

    # Normalizing the amplitudes
    max_amp = max(smoothed_amp)

    smoothed_amp = smoothed_amp / max_amp

    smoothed = smoothed_amp * np.exp(1j*np.angle(smoothed_phase))

    # with open("samplitude.txt", "w") as samplitude_f:
    #     for val in smoothed_amp:
    #         samplitude_f.write("{}\n".format(val))
    # with open("sphase.txt", "w") as sphase_f:
    #     for val in smoothed_phase:
    #         sphase_f.write("{}\n".format(np.angle(val)))
    smoothed_amp_phase = np.stack((np.absolute(smoothed_amp),np.angle(smoothed_phase)),axis=1)

    print(smoothed_amp_phase.shape)


    np.save("H_START_S.npy",smoothed_amp_phase)

    # Test interpolated pulse against the more realistic simulation

    result = simulate_more_realistic_qubit(duration=duration, values=smoothed, shots=1024, repetitions=1)

    realized_h_gate = result["unitary"]
    s_h_error = error_norm(realized_h_gate, ideal_h_gate)

    h_measurements = result["measurements"]
    h_probability, h_standard_error = estimate_probability_of_one(h_measurements)

    print("Realised Smoothed H Gate:")
    print(realized_h_gate)
    print("Ideal H Gate:")
    print(ideal_h_gate)
    print("Smoothed H Gate Error: " + str(s_h_error))

    # In[9]:

    # Normalizing the amplitudes
    absolutes = []
    for val in optimized_values:
        absolutes += [np.absolute(val)]
    max_amp = max(absolutes)

    # Write parameters to file

    # with open("amplitude.txt", "w") as amplitude_f:
    #     for val in absolutes:
    #         amplitude_f.write("{}\n".format(val / max_amp))
    # with open("phase.txt", "w") as phase_f:
    #     for val in optimized_values:
    #         phase_f.write("{}\n".format(np.angle(val)))

    unsmoothed_amp_phase = np.stack((absolutes,np.angle(optimized_values)),axis=1)

    print(unsmoothed_amp_phase.shape)

    np.save("H_START_U.npy",unsmoothed_amp_phase)

    return (h_error,s_h_error)

if __name__ == '__main__':
    run_main_h()
