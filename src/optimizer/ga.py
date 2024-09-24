import numpy as np
import pandas as pd

class GeneticAlgorithm():
    """
    This is an abstract implementation of the Genetic Algorithm
    """
    pool_size: int
    selection_rate: float # percentage of the population to be selected for crossover
    num_best: int # how many top individuals concerned
    best_records: pd.DataFrame # records of the best individuals
    iterations_done: int
    fitness_scores: np.ndarray
    iterations: int
    terminate_signal: bool = False

    def _initialise_population(self):
        raise NotImplementedError

    def fitness(self):
        """
        This function calculates the fitness of each individual in the population.
        """
        raise NotImplementedError
    
    def logging(self):
        raise NotImplementedError
    
    def selection(self):
        # select the top selection_rate percentage of the population
        num_selected = int(self.pool_size * self.selection_rate)
        selected_indices_sorted = np.argsort(self.fitness_scores)[::-1][:num_selected]
        self.population = self.population[selected_indices_sorted]
        self.fitness_scores = self.fitness_scores[selected_indices_sorted]
        # now things are selected and sorted
        
        # record the best individuals
        self.logging()

    def generate_offspring(self):
        """
        This function generates offspring from the selected individuals.
        """
        new_offspring = []
        while len(self.population) + len(new_offspring) < self.pool_size:
            a, b = np.random.choice(len(self.population), 2, replace=True)
            pending_mutation = [
                self.crossover(self.population[a], self.population[b]), self.crossover(self.population[b], self.population[a])
            ]
            mutated = [self.mutation(x) for x in pending_mutation]
            new_offspring.extend(mutated)
        self.population = np.concatenate((self.population, new_offspring))

    def crossover(self, a, b):
        raise NotImplementedError
    
    def mutation(self, x):
        raise NotImplementedError
    
    def step(self):
        """
        This function performs one step of the genetic algorithm.
        """
        self.fitness()
        self.selection()
        self.generate_offspring()
        self.iterations_done += 1

    def run(self):
        """
        This function runs the genetic algorithm for self.iterations iterations.
        """
        for _ in range(self.iterations):
            self.step()
            if self.terminate_signal:
                break