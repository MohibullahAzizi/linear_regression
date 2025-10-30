##################################################################
# Mohibollah Azizi
# DNA Center Point Problem - Local Search Algorithms              #
# Implemented: Hill Climbing (Descent) & Simulated Annealing      #
# Author: [Your Name]                                             #
# Date: [Date]                                                    #
##################################################################

import random
import numpy as np
import matplotlib.pyplot as plt
import time

################################
# Global Alphabet Definition  ##
################################
Alphabet = np.array(['a', 'c', 'g', 't'])

##############################
# Initialize a Random State ##
##############################
def initialize_state(n):
    ##################################################################
    # [inputs]                                                       #
    # length of the vector                                           #
    # [outputs]                                                      #
    # returns a random vector with length n and                      #
    # uniform probability from Alphabet elements as a numpy array    #
    ##################################################################
    Alphabet = np.array(['a', 'c', 'g', 't'])
    return np.random.choice(Alphabet, size=n)

#######################
# Evaluation Function #
#######################
def calculate_evaluation(genomes, state):
    ##################################################################
    # [inputs]                                                       #
    # genomes is a 2D numpy array and represents the set of DNAs     #
    # state is a 1D numpy array which represents "a" in the equation #
    # [outputs]                                                      #
    # the function returns the value of f(a)                         #
    ##################################################################
    distances = np.sum(genomes != state, axis=1)
    return np.max(distances)

################################################
# Neighboring State (Single Random Neighbor)   #
################################################
def get_neighbor_state(state):
    ##################################################################
    # [inputs]                                                       #
    # state is a 1D numpy array which represents a state             #
    # [outputs]                                                      #
    # a neighboring state of input is returned                       #
    # [notice]                                                       #
    # note the same state as input should not be returned            #
    # and the neighbor should be selected uniformly at random        #
    ##################################################################
    Alphabet = np.array(['a', 'c', 'g', 't'])
    neighbor = state.copy()
    i = np.random.randint(len(state))
    possible = [x for x in Alphabet if x != state[i]]
    neighbor[i] = np.random.choice(possible)
    return neighbor

#####################################
# Generate All Neighboring States   #
#####################################
def get_all_neighbors(state):
    ##################################################################
    # This function should generate and return a list of all         #
    # possible neighbors for a given state. A neighbor is defined    #
    # as a sequence that differs from the original state by exactly  #
    # one character.                                                 #
    # [inputs]                                                       #
    #  - state: a 1D numpy array representing the current state      #
    # [outputs]                                                      #
    #  - a list of all neighboring states                            #
    ##################################################################
    Alphabet = np.array(['a', 'c', 'g', 't'])
    neighbors = []
    for i in range(len(state)):
        for base in Alphabet:
            if base != state[i]:
                neighbor = state.copy()
                neighbor[i] = base
                neighbors.append(neighbor)
    return neighbors

#############################################
# Hill Climbing Algorithm (Descent Version) #
#############################################
def hill_climbing_descent(genomes, initial_state, max_iterations=1000):
    ##################################################################
    # Implement the Hill Climbing (descent) algorithm.               #
    # [inputs]                                                       #
    #  - genomes: a 2D numpy array representing the set of DNAs      #
    #  - initial_state: the state to start with                      #
    #  - max_iterations: a limit to prevent infinite loops           #
    # [outputs]                                                      #
    #  - best state found as a numpy array                           #
    #  - best state's evaluation value                               #
    #  - all the evaluations of current state as a normal list       #
    ##################################################################
    current_state = initial_state.copy()
    current_value = calculate_evaluation(genomes, current_state)
    evaluations = [current_value]

    for _ in range(max_iterations):
        neighbors = get_all_neighbors(current_state)
        best_neighbor = None
        best_value = current_value

        for n in neighbors:
            val = calculate_evaluation(genomes, n)
            if val < best_value:
                best_value = val
                best_neighbor = n

        if best_neighbor is not None:
            current_state = best_neighbor
            current_value = best_value
            evaluations.append(current_value)
        else:
            break

    return current_state, current_value, evaluations

##################################
# Simulated Annealing Algorithm  #
##################################
def simulated_annealing(genomes, initial_state, alpha, initial_temp, max_iteration, min_temperature):
    ##############################################################################
    # [inputs]                                                                   #
    # genomes is a 2D numpy array and represents the set of DNAs                 #
    # initial state is the state to start with                                   #
    # alpha is the temperature decay rate                                        #
    # initial temp is T0                                                         #
    # max_iteration is the maximum number of iteration (termination condition)   #
    # min_temperature is the minimum temperature (termination condition)         #
    # [outputs]                                                                  #
    # best state found as a numpy array                                          #
    # best state's evaluation value                                              #
    # all the evaluations of current state as a normal list                      #
    ##############################################################################
    curr_state = initial_state.copy()
    curr_value = calculate_evaluation(genomes, curr_state)
    best_state = curr_state.copy()
    best_value = curr_value
    temperature = initial_temp
    evaluations = [curr_value]

    for i in range(max_iteration):
        if temperature < min_temperature:
            break

        neighbor = get_neighbor_state(curr_state)
        neighbor_value = calculate_evaluation(genomes, neighbor)
        delta = neighbor_value - curr_value

        if delta < 0:
            curr_state = neighbor
            curr_value = neighbor_value
        else:
            probability = np.exp(-delta / temperature)
            if np.random.rand() < probability:
                curr_state = neighbor
                curr_value = neighbor_value

        if curr_value < best_value:
            best_state = curr_state.copy()
            best_value = curr_value

        evaluations.append(curr_value)
        temperature *= alpha

    return best_state, best_value, evaluations

############################################
# Helper Functions (Brute Force Utilities) #
############################################
def go_to_next(number, alphabet_length):
    idx = len(number) - 1
    while idx >= 0:
        if number[idx] < alphabet_length - 1:
            number[idx] += 1
            return True
        else:
            number[idx] = 0
            idx -= 1
    return False

def brute_force(genomes):
    n = len(genomes[0])
    curr_state_index = np.zeros(n).astype(int)
    best_state = None
    best_value = float('inf')

    while True:
        new_result = calculate_evaluation(genomes, Alphabet[curr_state_index])

        if new_result < best_value:
            best_value = new_result
            best_state = Alphabet[curr_state_index]

        if not go_to_next(curr_state_index, len(Alphabet)):
            break

    return best_state, best_value

##########################
# Running All Algorithms #
##########################
genomes_array = [
    np.array([['g', 'c', 'a', 't', 'c'],
              ['g', 'a', 'c', 't', 'c'],
              ['c', 'a', 'c', 'g', 'c']]),
    np.array([['a', 'c', 'g', 'g', 'g', 'a', 'c'],
              ['a', 'g', 'g', 'c', 'g', 'a', 'g'],
              ['c', 'g', 'g', 'g', 'g', 't', 'c']]),
    np.array([['c', 'c', 'a', 'c', 't', 'a', 'g', 'c', 'a'],
              ['c', 't', 'a', 'g', 't', 'c', 't', 'c', 't'],
              ['c', 't', 'c', 'c', 't', 'c', 'c', 'g', 'g']])
]

for genomes in genomes_array:
    initial_state = initialize_state(len(genomes[0]))

    start_time_sa = time.time()
    result_dna_sa, result_value_sa, _ = simulated_annealing(genomes, initial_state, 0.9, 500, 1000, 1e-3)
    time_sa = time.time() - start_time_sa

    start_time_hc = time.time()
    result_dna_hc, result_value_hc, _ = hill_climbing_descent(genomes, initial_state)
    time_hc = time.time() - start_time_hc

    start_time_bf = time.time()
    result_dna_bf, result_value_bf = brute_force(genomes)
    time_bf = time.time() - start_time_bf

    print(f'Simulated Annealing found solution {result_dna_sa} with value {result_value_sa} in {time_sa*1000:.3f} milliseconds')
    print(f'Hill Climbing       found solution {result_dna_hc} with value {result_value_hc} in {time_hc*1000:.3f} milliseconds')
    print(f'Brute Force         found solution {result_dna_bf} with value {result_value_bf} in {time_bf*1000:.3f} milliseconds')
    print("-" * 50)

######################################
# Comparison Plot Between Algorithms #
######################################
example_genomes = np.array([['c', 'c', 'a', 'c', 't', 'a', 'g', 'g', 'a'],
                            ['c', 't', 'a', 'g', 't', 'c', 't', 'g', 'a'],
                            ['c', 't', 'c', 'c', 't', 'c', 'c', 'g', 'a']])

initial_state = initialize_state(len(example_genomes[0]))
_, _, evaluations_sa = simulated_annealing(example_genomes, initial_state, 0.95, 500, 1000, 1e-3)
_, _, evaluations_hc = hill_climbing_descent(example_genomes, initial_state)

plt.plot(evaluations_sa, label='Simulated Annealing')
plt.plot(evaluations_hc, label='Hill Climbing')
plt.title('Algorithm Comparison')
plt.ylabel('Evaluation Value (Radius)')
plt.xlabel('Iteration')
plt.legend()
plt.grid(True)
plt.show()

##########################
# Draw Results Helper    #
##########################
def draw_results(evaluations):
    ############################
    # no need to the any thing #
    ############################
    plt.plot(evaluations)
    plt.title('Simulated Annealing algorithm')
    plt.ylabel('value')
    plt.xlabel('iteration')
    plt.show()

##########################################
# Cooling Rate (alpha) Effect Experiment #
##########################################
for alpha in [0.8, 0.9, 0.95]:
    result_dna, result_value, evaluations = simulated_annealing(example_genomes, initial_state, alpha, 500, 1000, 1e-3)
    print(f'results with α={alpha}: {result_dna} and radius: {result_value}')
    draw_results(evaluations)
