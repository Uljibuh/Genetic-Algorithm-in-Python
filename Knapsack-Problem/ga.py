from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy


import matplotlib.pyplot as plt 
import seaborn as sns 

import knapsack

# problem constants
# create the knapsack problem instance to be used:
knapsack = knapsack.Knapsack01Problem()

# genetic algorithm constants
POPULATION_SIZE = 50
P_CROSSOVER = 0.9 # probability for crossover
P_MUTATION = 0.1  # probability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 1


# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()


# create an operator that randomly return 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximiing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))    

# craate the iindividual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax) 

# crate the individual operator to fill up individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(knapsack))

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)    


# fitness calculation
def knapsackValue(individual):
    return knapsack.getValue(individual), # return a tuple

toolbox.register("evaluate", knapsackValue)


# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize=3)

# single point crossover:
toolbox.register("mate", tools.cxTwoPoint)

# flip-bit mutation
# indpb: independent proability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(knapsack))


# genetic algorithm flow

def main():
    
    # craete initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    
    
    # define the hall of fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    
    # perform the genetic algorithm flow with hof feature added:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,verbose=True)
    
    # print best solution found
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    
    print("-- Knapsack Items = ")
    knapsack.printItems(best)
    
    # extract statistics
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    
    # plot statisitcs
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Avarage Fitness')
    plt.title('Max and average fitness over generations')
    plt.show()
    

if __name__ == "__main__":
    main()    