# 🌻 **Meliora** 🌻 - **Applied Quantum Learning Control**

## 🟣 **Our Team**
We are a team of 4 individuals from the **University of Rochester**, named after our school motto, meaning *"Ever Better"*.

**|00〉** Ryan Cocuzzo

**|01〉** Daniel Busaba

**|10〉** Michael Taylor

**|11〉** Dominick Harasimiuk


<br>
<hr>


## 🟣 **Get Started**


To build the program, simply 

  (1) Download the code 
  
  (2) Run the following from the terminal (root):

    cd production_code
    python ./pipeline.py

  (3) If you want to run JSON output on the real
  qubit, do the following (from root):

    cd production_code
    python ./run_json.py [/path/to/json/file]
    python ./run_json.py ../h_gate_pulse.json



## 🟣 **Introduction**

### **Hadamard Gate Pulse**
![HADAMARD](./Figures/H_Pulse_best.png)

We sought to design *custom pulse envelope functions* to deterministically apply logic gates to a superconducting Q-CTRL qubit. Specifically, we designed a pulse that is robust to environmental factors to implement a NOT logical gate and a Hadamard logical gate using microwave pulses on a real qubit system. To accomplish this, we took a two step approach consisting of both *Quantum Optimum Control* and *Quantum Learning Control methods*, balancing the shortfalls of each to find the optimum pulse function. Quantum Optimum Control is quick and efficient; however, it requires that the programmers know the Hamiltonian of the system very well to accomplish a sufficient degree of success. On the other hand, Quantum Learning Control does not use any knowledge of the system, relying on a *Genetic Machine Learning algorithm* to converge on a viable system. Consequentially, if given a poor starting point, it may not converge well. By first calculating a modified Quantum Optimum Control pulse from a semi-realistic model (first, using Q-CTRL's provided "Simulating a More Realistic Superconducting Qubit" example), we can find a very good starting point for the Quantum Learning Control method to then optimize into a viable pulse structure for a given pulse. **For more info, check out our Challenge Summary.**

### 🟣 **NOT Gate Pulse**
![NOT](./Figures/NOT_Pulse_best.png)


</div>
<hr>

## 🟣 **First Step: Quantum Optimum Control Optimization with Sinc Interpolation**
We were quickly able to generate a pulse that would easily satisfy the model. The issue though was this proposed pulse at scale is not very physically viable. Specifically, it would have a phase and amplitude that oscillated at incredibly fast rates. After experimenting with various different smoothing algorithms and parameters for the optimization, we settled on a method that would produce a more physically feasible pulse shape. We did this with a combination of two modifications to the method discussed in the Twitch stream. First, we changed the definition of the amplitude function in the optimization from a `bounded_optimization_variable` to an `anchored_difference_bounded_variables` and added a `difference_bound` constraint. This ensured that between the adjacent segments, the amplitude of the pulse would not change by more than a small tunable parameter. This made the pulse amplitude function more realistic and well behaved. However, this still left the phase as rapidly oscillating and the amplitude as having sharp peaks/troughs. To fix this, we purposely had the optimizer optimize over a relatively small number of segments and then upscaled it by using a sinc (aka sin(pi * x) / (pi * x)) function interpolation algorithm onto the amplitude. This smoothed out the peaks and valleys of the amplitude function, making it a more natural loooking function. The rapidly changing phase problem is also mitigated by this because the phase is simply copied n times for an upscaling factor of n. This increases the amount of time that the microwave pulse is at a given phase. To validate that the interpolated pulse function is still valid for the more realistic simulation, we plug the smoothed pulse back into this simulation and ensure that the error is still relatively small. By tuning the number of segments, the factor of upscaling and the amplitude difference_bound constraint, we reliably errors on the real qubit within **~5-10%**. This provided us with a good starting point to plug into our Quantum Learning Control Using Genetic Algorithm Postprocessing

<hr>

## 🟣 **Second Step: Quantum Learning Control Using Genetic Algorithm Postprocessing**

After being pre-processed on a simulated Q-bit, the generated microwave is then optimized using a genetic algorithm. At the heart of the genetic algorithm is the loss or fitness function that evaluates how close the microwave is to making the Q-bit behave as desired. We defined this function as the real mean squared of the expected and actual Q-bit value probabilities divided by the repetition count. We divide by the repetition count to provide some penalty for signals with error but to maintain priority on an accurate first shot.



## 🟣 **Results and Conclusions**

We were able to successfully model both our target gates with a diminunative amount of loss **(~5-10)**. Both of our high-fidelity gates implement cutting-edge
ML techniques, built by hand. Our library is well-defined and our models converge to any reasonable limits suggested by the Heisenburg Uncertainty Principle. 

For a quick summary on our gate performance here, our Hadamard gate was able to achieve probabilities between `.47` to `.52` for states 0,1 for any number of repetitions,
up to and including `64` repetitions! The NOT gate was more challenging because it is inherently hard to transfer an entire population from one state to another completely (as opposed to the $H$ gate which is more entropically favorable to converging).

We were able to achieve probabilities between `.79` and `.90` reliably through `16` repetitions.

**NOT** Gate Results:
```
control:             P(|0>) = 0.09, P(|1>) = 0.90, P(|2>) = 0.00
With  1 repetitions: P(|0>) = 0.13, P(|1>) = 0.87, P(|2>) = 0.00
With  4 repetitions: P(|0>) = 0.10, P(|1>) = 0.90, P(|2>) = 0.00
With 16 repetitions: P(|0>) = 0.21, P(|1>) = 0.79, P(|2>) = 0.00
With 32 repetitions: P(|0>) = 0.40, P(|1>) = 0.60, P(|2>) = 0.00
With 64 repetitions: P(|0>) = 0.07, P(|1>) = 0.92, P(|2>) = 0.01
LOSS LIST:  [0.03373908996582031, 0.00481724739074707, 0.005485117435455322, 0.010044366121292114, 0.00017990171909332275]
BEST LOSS:  0.05426572263240814
```
![NOT](./Figures/NOT_Pulse_best.png)

**Hadamard** Gate Results:
```
control:             P(|0>) = 0.49, P(|1>) = 0.50, P(|2>) = 0.00
With  1 repetitions: P(|0>) = 0.52, P(|1>) = 0.48, P(|2>) = 0.00
With  4 repetitions: P(|0>) = 0.50, P(|1>) = 0.50, P(|2>) = 0.00
With 16 repetitions: P(|0>) = 0.49, P(|1>) = 0.51, P(|2>) = 0.00
With 32 repetitions: P(|0>) = 0.50, P(|1>) = 0.48, P(|2>) = 0.01
With 64 repetitions: P(|0>) = 0.50, P(|1>) = 0.47, P(|2>) = 0.02
LOSS LIST:  [0.0008411407470703125, 3.0994415283203125e-06, 1.9371509552001953e-05, 1.0401010513305664e-05, 1.0132789611816406e-05]
BEST LOSS:  0.0008841454982757568
```
![HADAMARD](./Figures/H_Pulse_best.png)

## Other Notes

API Limitations:
    256 segments
    64 repititions
