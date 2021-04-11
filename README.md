# qctrl-qchack-meliora

## Team Information
We are a team of 4 individuals from the University of Rochester:

- Ryan Cocuzzo
- Daniel Busaba
- Michael Taylor
- Dominick Harasimiuk

Our team name is "meliora" - this is the motto of our school and it means "ever better".

## Introduction



### First Step: Quantum Optimum Control Optimization with Sinc Interpolation
The scripts _____ and ______ use a similar approach to the twitch webinar to utilize the more realistic model of the qubit to create an optimal pulse shape. We found that while it was fairly trivial to generate a pulse that would easily satisfy the model, this proposed pulse frequently not very physically viable. Specifically, it would have a phase and amplitude that oscillated at incredibly fast rates. After experimenting with various different smoothing algorithms and parameters for the optimization, we settled on a method that would produce a more physically feasible pulse shape. We did this with a combination of two modifications to the method demonstrated in the Twitch stream. First, we changed the definition of the amplitude function in the optimization from a bounded_optimization_variable to an anchored_difference_bounded_variables and added a difference_bound constraint. This ensured that between the adjacent segments, the amplitude of the pulse would not change by more than a small tunable parameter. This made the pulse amplitude function more realistic and well behaved. However, this still left the phase as rapidly oscillating and the amplitude as having sharp peaks/troughs. To fix this, we purposely had the optimizer optimize over a relatively small number of segments and then upscaled it by using a sinc (aka sin(pi * x) / (pi * x)) function interpolation algorithm onto the amplitude. This smoothed out the peaks and valleys of the amplitude function, making it a more natural loooking function. The rapidly changing phase problem is also mitigated by this because the phase is simply copied n times for an upscaling factor of n. This increases the amount of time that the microwave pulse is at a given phase. To validate that the interpolated pulse function is still valid for the more realistic simulation, we plug the smoothed pulse back into this simulation and ensure that the  

## Second Step: Quantum Learning Control Using Genetic Algorithm Postprocessing


## Results and Conclusions

## Other Notes

API Limitations:
    256 segments
    64 repititions

