import numpy as np
import matplotlib.pyplot as plt

def compute_loss(p1: float, gate: str):
    if gate == 'H'
        return p1
    return 0.5 - abs(0.5-p1)

def experiment_results_to_probabilities(experiment_results: list, reps: int, shot_count: int):
    return [(
        p0 = measurement_counts.count(0) / shot_count,
        p1 = measurement_counts.count(1) / shot_count,
        p2 = measurement_counts.count(2) / shot_count
    ) for repetition_count, measurement_counts in zip( reps, experiment_results.measurements)]


def real_to_complex(amplitude, phase, values):

    return amplitudes * np.exp(1j * phases)
    controls.append({"duration": duration, "values": values})

# @param x  [(amplitude,phase), ... N times]
#            ...
#            P times 
#            where N = Segment count length
#                  P = population size
def fitness_func(x: np.array, N: int, P: int):
    
    for batch in x:

        for item in batch:
            
            amplitude = item[0]
            phase     = item[1]
            values    = item[2]

            return real





    
    


    
#selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # print("P1:", p1, len(p1))
    # print("P2", p2)
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # print("C1:", c1)
    # print("C2:", c2)

    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1))
        # perform crossover
        c1 = np.concatenate((p1[:pt,:], p2[pt:,:]),axis=0)
        c2 = np.concatenate((p2[:pt,:], p1[pt:,:]),axis=0)
    return [c1, c2]
 
# mutation operator
def mutation(values, r_mut):
    for i in range(len(values)):
        # check for a mutation
        if np.random.rand() < r_mut:
            # flip the bit
            values[i][0] = values[i][0] + np.random.normal(0,1)
            values[i][1] = values[i][1] + np.random.normal(0,1)

def gaussian_search(objective, n_iter, pop, r_cross, r_mut):
    # keep track of best solution
    best, best_eval = 0, objective(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i].flatten(), scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(len(pop))]
        #print(selected)



        # create the next generation
        children = list()
        for i in range(0, len(pop), 2): #step 2 so can do i+1
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

n_pop = 10
segment_count = 4
pop = np.random.rand(n_pop,segment_count,2)
n_iter = 10
r_cross = 0.9
r_mut = 1/(segment_count*2)

best, score = gaussian_search(fitness_func, n_iter, pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))

# r_cross = 0.5
# p1 = np.zeros((4,2))
# p2 = np.ones((4,2))
# print("Cross: ", crossover(p1, p2, r_cross))
