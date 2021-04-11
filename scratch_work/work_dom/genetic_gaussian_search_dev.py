import numpy as np
import matplotlib.pyplot as plt

#adapted from: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

def np_2d_print(arr):
    amps = str(arr[:,0])
    phases = str(arr[:,1])
    final_str = "\n Amplitudes: " + amps + "\n Phases: " + phases
    return final_str

def loss_func(x):
    return x.sum()

#selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = np.random.randint(len(pop)) #select from population
    for ix in np.random.randint(0, len(pop), k-1): #select a pair of indices from the population
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, p_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < p_cross:
        # select crossover point
        pt = np.random.randint(1, len(p1)) 
        # perform crossover
        c1 = np.concatenate((p1[:pt,:], p2[pt:,:]),axis=0)
        c2 = np.concatenate((p2[:pt,:], p1[pt:,:]),axis=0)
    return [c1, c2]
 
# mutation operator
def mutation(values, p_mut_amp, p_mut_phase, gaussian_params):
    for i in range(len(values)): #for every index in vector
        # check for a mutation
        if np.random.rand() < p_mut_amp:
            values[i][0] = values[i][0] + np.random.normal(
                gaussian_params["amp_mean"],
                gaussian_params["amp_sd"]
            )
        if np.random.rand() < p_mut_phase:
            values[i][1] = values[i][1] + np.random.normal(
                gaussian_params["ph_mean"],
                gaussian_params["ph_sd"]
            )

            #check for  OOB - Amplitude
            if values[i][0] > 1: values[i][0] = 1
            if values[i][0] < 0: values[i][0] = 0

            #check for OOB - Phase
            if values[i][1] > 2*np.pi: 
                values[i][1] = values[i][1] - np.floor(values[i][1]/(2*np.pi))*2*np.pi
            if values[i][1] < 0: 
                values[i][1] = values[i][1] - np.floor(values[i][1]/(2*np.pi))*2*np.pi


def genetic_gaussian_search(loss, seed, params):

    #extract parameters
    n_pop = params["population size"]
    n_iter = params["iterations"]
    p_cross = params["crossover prob"]
    p_mut_amp = params["mutation prob amp"]
    p_mut_phase = params["mutation prob phase"]

    #build population
    pop = np.repeat(seed[np.newaxis,:,:], n_pop, axis=0)

    # keep track of best solution
    best, best_eval = 0, loss(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [loss(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s \n) = %.3f" % (gen,  np_2d_print(pop[i]), scores[i]))

        # select parents
        selected = [selection(pop, scores) for _ in range(len(pop))]
        #there are always len(pop) number of parents

        # create the next generation
        children = list()
        for i in range(0, len(pop), 2): #step 2 so can do i+1
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, p_cross):
                # mutation
                mutation(c, p_mut_amp, p_mut_phase, params["gaussian_params"])
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]


if __name__ == '__main__':
    segment_count = 4
    seed = np.random.rand(segment_count,2)
    params = {
        "gaussian_params" : {
            "amp_mean": 0,
            "ph_mean": 0,
            "amp_sd": 1,
            "ph_sd": 1,
        },
        "iterations": 100,
        "population size": 10,
        "crossover prob": 0.9,
        "mutation prob amp": 1/(segment_count*2),
        "mutation prob phase": 1/(segment_count*2)
    }

    best, score = genetic_gaussian_search(loss_func, seed, params)
    print('Done! THE BEST IS:')
    print('f(%s \n) = %f' % (np_2d_print(best), score))