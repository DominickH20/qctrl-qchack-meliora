#!/usr/bin/env python
# coding: utf-8


import sim_h
import sim_not
import numpy as np
import real_q


from dotenv import dotenv_values
config = dotenv_values(".env")
email=config['EMAIL']
password=config['PW']

def get_best():
  """ Returns the most promising initial wave lengths for the NOT and H gates. """

  best_h = 1
  best_not = 1
  runs = 3

  np.save("H_START_S_BEST.npy",0)
  np.save("NOT_START_S_BEST.npy",0)

  for i in range(runs):
    
    # Run Sim-H
    sim_h.run_main_h()[0]
    # Run Sim-NOT
    sim_not.run_main_not()[0]

    hloss = run_waves("H")
    print("Loss of H Gate")
    print(hloss)
    nloss = run_waves("N")
    print("Loss of NOT Gate")
    print(nloss)

    # If this is a better H value, update
    if (hloss < best_h):
      best_h = hloss
      overwriteH()

    # If this is a better NOT value, update
    if (nloss < best_not):
      best_not = nloss
      overwriteNOT()


  h_waves_list = np.load("H_START_S_BEST.npy")
  not_waves_list = np.load("NOT_START_S_BEST.npy")

  return (h_waves_list, not_waves_list)

def run_waves(circuit):
    # Make sure circuit name is valid and load data
    circuit = circuit.upper()
    if circuit == "H":
        waves_list = np.load("H_START_S.npy")
    elif circuit == "N":
        waves_list = np.load("NOT_START_S.npy")
    else:
        print("Invalid circuit ID: {}".format(circuit))
        exit(1)

    # Define parameters for run
    max_drive_amplitude = 2 * np.pi * 20                       # MHz
    params = {
        "segment_count": waves_list.shape[0],
        "duration": 5 * np.pi / (max_drive_amplitude) * 1000,  # Convert to ns
        "shot_count": 1024,
    }

    # Run on the quantum computer
    controls = [waves_list, waves_list, waves_list] #P * N * 2
    
    repetitions, experiment_results = real_q.run_on_q(controls, params)

    repetitions = np.split(np.array(repetitions), len(controls))
    measurements = np.split(np.array(experiment_results.measurements), len(controls))
    losses = []

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
            if circuit == "H":
                loss_sum += ((p0 - 0.5) ** 2) / repetition_count
                loss_sum += ((p1 - 0.5) ** 2) / repetition_count
            elif circuit == "N":
                loss_sum += ((p0 - 0) ** 2) / repetition_count
                loss_sum += ((p1 - 1) ** 2) / repetition_count
        losses += [loss_sum]

    return loss_sum  

def overwriteNOT():
  # open both files

  temp_not = np.load('NOT_START_S.npy')
  np.save("NOT_START_S_BEST.npy",temp_not)

def overwriteH():
  # open both files
  temp_h = np.load('H_START_S.npy')
  np.save("H_START_S_BEST.npy",temp_h)
  
get_best()