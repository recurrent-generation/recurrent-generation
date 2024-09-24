# append project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.optimizer.ga import GeneticAlgorithm
from src.llm.hookedLLM import HookedLLM

import numpy as np
import time
import pandas as pd

from tqdm import tqdm

FITNESS_CRITERION = ['token_count']

class GeneticAlgorithm4LLM(GeneticAlgorithm):
    """
    This class implements a genetic algorithm for attacking large language models (LLMs),
    with the same methodology used in the Sponge Examples work.
    """
    def __init__(self, pool_size: int, iterations: int, llm: HookedLLM, input_len: int, 
                 selection_rate: float, mutation_ratio: float,
                 log_path: str, num_best: int, 
                 fitness_criteria: str, eval_times: int = 1, batch_size: int = 1, terminate_if_found: bool=False, **kwargs) -> None:
        # general
        self.pool_size = pool_size
        self.iterations = iterations
        self.iterations_done = 0
        self.selection_rate = selection_rate
        self.log_path = log_path
        self.num_best = num_best
        # initialize the best_records dataframe with columns: 'iteration', 'best_results', 'average_fitness'
        # 'best_results' should be a dict mapping 'prompt' to 'fitness'
        self.best_records = pd.DataFrame(columns=['iteration', 'best_results', 'average_fitness', 'max_tokens_generated'])
        # the goal is to hit the max_new_tokens limit
        self.alg_log = pd.DataFrame(columns=['first_iteration_to_hit_goal', 'time_to_hit_goal'])
        self.start_time = time.time()

        # specific to LLM
        self.llm = llm
        self.dict_size = llm.get_vocab_size()
        self.input_len = input_len
        self.mutation_ratio = mutation_ratio
        self.fitness_criteria = fitness_criteria
        self.eval_times = eval_times
        self.batch_size = batch_size
        # initialize the experiences dataframe with columns: 'iteration', 'prompt', 'response', 'fitness'
        self.experiences = pd.DataFrame(columns=['iteration', 'prompt', 'response', 'fitness', 'tokens_generated', 'time_taken', 'time_per_token'])

        self._initialise_population()

        self.goal_reached = False
        self.terminate_if_found = terminate_if_found

    def _initialise_population(self):
        # population is a np array of shape (pool_size, input_len), with random integers from 0 to dict_size representing the tokens
        self.population = np.random.randint(0, self.dict_size, (self.pool_size, self.input_len))

    def fitness(self) -> np.ndarray:
        """
        The longer it takes the model to generate the output, the better.
        """
        assert self.eval_times > 0
        if self.eval_times > 1:
            assert self.fitness_criteria in ['token_count'] # current impl
        fitness_scores = np.zeros(self.pool_size)
        self.max_tokens_generated = np.zeros(self.pool_size)
        print('Calculating fitness scores for iteration', self.iterations_done)
        for i, individual in tqdm(enumerate(self.population), total=len(self.population)):
            if self.eval_times == 1:
                start_time = time.time()
                # run
                prompt = self.llm.decode(individual)
                response = self.llm.chat(prompt)[0]
                # monitor teardown
                time_cost = time.time() - start_time
                response_max_token_count = self.llm.get_num_tokens(response)

                fitness_scores[i] = {
                    'token_count': response_max_token_count
                }[self.fitness_criteria]

                responses_token_count = [response_max_token_count]

                response = [response] # to match the format of the experiences dataframe
            else:
                # fitness is average token_count
                assert self.eval_times % self.batch_size == 0
                num_evals = self.eval_times // self.batch_size
                response = []
                for j in range(num_evals):
                    prompt_batch = [self.llm.decode(individual) for _ in range(self.batch_size)]
                    response_batch = self.llm.chat(prompt_batch)
                    response.extend(response_batch)

                responses_token_count = [self.llm.get_num_tokens(response) for response in response]

                response_max_token_count = np.max(responses_token_count)

                fitness_scores[i] = np.mean(responses_token_count)

            self.max_tokens_generated[i] = response_max_token_count
            new_exp = {
                'iteration': self.iterations_done,
                'prompt': [self.llm.decode(individual)],
                'response': response,
                'fitness': fitness_scores[i],
                'tokens_generated': responses_token_count,
            }
            if self.eval_times == 1:
                new_exp['time_taken'] = time_cost
                new_exp['time_per_token'] = time_cost / response_max_token_count
            self.experiences.loc[len(self.experiences)] = new_exp
            # export the experiences dataframe to a csv file
            self.experiences.to_csv(os.path.join(self.log_path, 'experiences.csv'))
        self.fitness_scores = fitness_scores

    def logging(self):
        # logging is called at selection, by then the population is selected and the selected instances are sorted according to the fitness scores
        # append to best records and export
        best_results = []
        for i in range(self.num_best):
            # best_results[self.llm.decode(self.population[i])] = self.fitness_scores[i]
            best_results.append((self.llm.decode(self.population[i]), self.fitness_scores[i]))
        self.best_records.loc[len(self.best_records)] = {
            'iteration': self.iterations_done,
            'best_results': best_results,
            # average fitness score of the best concerned individuals
            'average_fitness': np.mean(self.fitness_scores[:self.num_best]),
            'max_tokens_generated': np.max(self.max_tokens_generated),
            'max_fitness': np.max(self.fitness_scores)
        }
        self.best_records.to_csv(os.path.join(self.log_path, 'best_records.csv'))
        # if the goal is hit, record the iteration and time
        if np.max(self.max_tokens_generated) >= self.llm.generation_configs['max_new_tokens']-20 \
            and not self.goal_reached:
                self.alg_log.loc[0] = {
                    'first_iteration_to_hit_goal': self.iterations_done,
                    'time_to_hit_goal': time.time() - self.start_time
                }
                self.alg_log.to_csv(os.path.join(self.log_path, 'alg_log.csv'))
                self.goal_reached = True
                
                # terminate the algorithm
                if self.terminate_if_found:
                    self.terminate_signal = True

    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # concatenate the first half of a and the second half of b
        split_point = self.input_len // 2
        return np.concatenate((a[:split_point], b[split_point:]))
    
    def mutation(self, x):
        # mutate self.mutation_ratio (round up) of the tokens
        num_mutations = int(np.ceil(self.input_len * self.mutation_ratio))
        mutation_indices = np.random.choice(self.input_len, num_mutations, replace=False)
        x[mutation_indices] = np.random.randint(0, self.dict_size, num_mutations)
        return x