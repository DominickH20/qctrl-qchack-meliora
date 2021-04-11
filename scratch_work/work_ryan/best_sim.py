#!/usr/bin/env python
# coding: utf-8


import sim_h
import sim_not
import numpy as np
import real_q

def get_best():
  """ Returns the most promising initial values for the NOT and H gates.

  """

  best_h = 1
  best_not = 0
  runs = 10
  for i in range(runs):
    
    # Run Sim-H
    sim_h.run_main_h(best_h)[0]
    # Run Sim-NOT
    sim_not.run_main_not(best_not)[0]

    hloss = run_waves("H")
    nloss = run_waves("N")

    # If this is a better H value, update
    if (abs(hloss - 0.5) < best_h):
      best_h = hloss
      overwriteH()

    # If this is a better NOT value, update
    if (n < best_n):
      best_not = nloss
      overwriteNOT()

  h_waves_list = np.load("H_START_S_BEST.npy")
  not_waves_list = np.load("NOT_START_S_BEST.npy")

  return (h_waves_list, not_waves_list)

def run_waves(circuit):
    # Make sure circuit name is valid and load data
    circuit = circuit.upper()
    if circuit == "H":
        waves_list = np.load("H_START.npy")
    elif circuit == "N":
        waves_list = np.load("N_START.npy")
    else:
        print("Invalid circuit ID: {}".format(circuit))
        exit(1)

    # Define parameters for run
    max_drive_amplitude = 2 * np.pi * 20                       # MHz
    params = {
        "segment_count": 64,
        "duration": 5 * np.pi / (max_drive_amplitude) * 1000,  # Convert to ns
        "shot_count": 10,
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
  with open('NOT_START_S.npy','r') as firstfile, open('NOT_START_S_BEST.npy','w') as secondfile:
        
    # read content from first file
    for line in firstfile:
              
      # write content to second file
      secondfile.write(line)

def overwriteH():
  # open both files
  with open('H_START_S.npy','r') as firstfile, open('H_START_S_BEST.npy','w') as secondfile:
        
    # read content from first file
    for line in firstfile:
              
      # write content to second file
      secondfile.write(line)
  
get_best()