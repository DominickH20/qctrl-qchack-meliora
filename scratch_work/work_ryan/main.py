#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import real_q

def compute_loss(p1, gate):
    if gate == 'H':
        return p1
    return 0.5 - abs(0.5-p1)

def experiment_results_to_probabilities(experiment_results, reps, shot_count):
    return [(
        measurement_counts.count(0) / shot_count,
        measurement_counts.count(1) / shot_count,
        measurement_counts.count(2) / shot_count
    ) for repetition_count, measurement_counts in zip( reps, experiment_results.measurements)]

def main(circuit):
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
    repetitions, experiment_results = real_q.run_on_q([waves_list, waves_list, waves_list], params)

    # Generate output value probabilities for results
    probabilities = experiment_results_to_probabilities(experiment_results=experiment_results, reps=repetitions, shot_count=params["shot_count"] )

    # Compute loss for output probabilities
    loss = [compute_loss(p1, circuit) for p0,p1,p2 in probabilities]
    
    return loss

# Entry point for program
if __name__ == "__main__":
    # Parse parameters
    p = argparse.ArgumentParser(description="A tool to generate signals to create quantum computer circuits")
    p.add_argument("circuit", help="The circuit to be optimized (H or N)")
    args = p.parse_args()

    main(args.circuit)
