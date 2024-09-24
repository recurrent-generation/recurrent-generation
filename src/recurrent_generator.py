# append project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import click
import yaml

@click.command()
@click.option('-exp_folder', help='Location of the experiment.')
def main(exp_folder):
    # Load the config file
    config_path = os.path.join(exp_folder, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    from src.llm.hookedLLM import HookedLLM
    from src.optimizer.ga4llm import GeneticAlgorithm4LLM
    if config['model']['sampling_params']['num_return_sequences'] != 1:
        raise ValueError('num_return_sequences must be 1 for the genetic algorithm to work.')
    hooked_llm = HookedLLM(**config['model'])

    ga = GeneticAlgorithm4LLM(llm=hooked_llm, log_path=exp_folder, **config['ga'])

    ga.run()

if __name__ == "__main__":
    main()