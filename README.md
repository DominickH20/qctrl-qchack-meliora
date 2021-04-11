# qctrl-qchack-meliora

## Team Information
We are a team of 4 individuals from the University of Rochester:

- Ryan Cocuzzo
- Daniel Busaba
- Michael Taylor
- Dominick Harasimiuk

Our team name is "meliora" - this is the motto of our school and it means "ever better".

## Introduction
In this hackathon challenge, we sought to design custom pulse envelope functions to deterministically apply logic gates to a superconducting Q-CTRL qubit. Specifically, we designed a pulse that is robust to environmental factors to implement a NOT logical gate and a Hadamard logical gate using microwave pulses on a real qubit system. To accomplish this feat, we took a two step approach consisting of both Quantum Optimum Control and Quantum Learning Control methods, balancing the shortfalls of each to find the optimum pulse function. Quantum Optimum Control is quick and efficient; however, it requires that the programmers know the Hamiltonian of the system very well to accomplish a sufficient degree of success. On the other hand, Quantum Learning Control does not use any knowledge of the system, relying on a Genetic Machine Learning algorithm to converge on a viable system. Consequentially, if given a poor starting point, it may not converge well. By first calculating a modified Quantum Optimum Control pulse from a semi-realistic model (model copied from the "Simulating a More Realistic Superconducting Qubit" documentation), we can find a very good starting point for the Quantum Learning Control method to then optimize into a viable pulse structure for a given pulse.


### First Step: Quantum Optimum Control Optimization with Sinc Interpolation
The scripts sim_h.py and sim_not.py use a similar approach to the twitch webinar to utilize the more realistic model of the qubit to create an optimal pulse shape for the Hadamard and NOT logical gates respectively. We found that while it was fairly trivial to generate a pulse that would easily satisfy the model, this proposed pulse frequently not very physically viable. Specifically, it would have a phase and amplitude that oscillated at incredibly fast rates. After experimenting with various different smoothing algorithms and parameters for the optimization, we settled on a method that would produce a more physically feasible pulse shape. We did this with a combination of two modifications to the method demonstrated in the Twitch stream. First, we changed the definition of the amplitude function in the optimization from a bounded_optimization_variable to an anchored_difference_bounded_variables and added a difference_bound constraint. This ensured that between the adjacent segments, the amplitude of the pulse would not change by more than a small tunable parameter. This made the pulse amplitude function more realistic and well behaved. However, this still left the phase as rapidly oscillating and the amplitude as having sharp peaks/troughs. To fix this, we purposely had the optimizer optimize over a relatively small number of segments and then upscaled it by using a sinc (aka sin(pi * x) / (pi * x)) function interpolation algorithm onto the amplitude. This smoothed out the peaks and valleys of the amplitude function, making it a more natural loooking function. The rapidly changing phase problem is also mitigated by this because the phase is simply copied n times for an upscaling factor of n. This increases the amount of time that the microwave pulse is at a given phase. To validate that the interpolated pulse function is still valid for the more realistic simulation, we plug the smoothed pulse back into this simulation and ensure that the error is still relatively small. By tuning the number of segments, the factor of upscaling and the amplitude difference_bound constraint, we reliably errors on the real qubit within ~5-10%. This provided us with a good starting point to plug into our Quantum Learning Control Using Genetic Algorithm Postprocessing

## Second Step: Quantum Learning Control Using Genetic Algorithm Postprocessing
After being pre-processed on a simulated Q-bit, the generated microwave is then optimized using a genetic algorithm. At the heart of the genetic algorithm is the loss or fitness function that evaluates how close the microwave is to making the Q-bit behave as desired. We defined this function as the real mean squared of the expected and actual Q-bit value probabilities divided by the repetition count. We divide by the repetition count to provide some penalty for signals with error but to maintain priority on an accurate first shot.

## Results and Conclusions

## Other Notes

API Limitations:
    256 segments
    64 repititions

