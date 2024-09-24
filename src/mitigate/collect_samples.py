# append project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import click

import pandas as pd
import yaml
from tqdm import tqdm


def collect_samples(src_folder: str, target_folder: str):
    print('collecting from', src_folder)
    # load config at src
    with open(f'{src_folder}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    THRESHOLD_FOR_LOOPS = config['model']['sampling_params']['max_new_tokens'] - 20
    # create target folder if not exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    NON_RECURRENT_COLLECTION_PATH = os.path.join(target_folder, 'non_recurrent_samples.csv')
    RECURRENT_SAMPLE_PATH = os.path.join(target_folder, 'recurrent_samples.csv')
    COLLECTION_PATHS = {
        'non_recurrent': NON_RECURRENT_COLLECTION_PATH,
        'recurrent': RECURRENT_SAMPLE_PATH
    }

    for collect_type in ['recurrent', 'non_recurrent']:
    # for collect_type in ['non_recurrent']:
    # for collect_type in ['recurrent']:
        collection_path = COLLECTION_PATHS[collect_type]

        single_exp_folder = src_folder
        if os.path.isdir(single_exp_folder):
            # print(single_exp_name)
            
            experiences_path = os.path.join(single_exp_folder, 'experiences.csv')
            experiences = pd.read_csv(experiences_path)

            try:
                all_samples = pd.read_csv(collection_path)
            except:
                # make a new one
                all_samples = pd.DataFrame(columns=['prompt', 'response'])

            # concat non-repeating prompts
            for i, row in tqdm(experiences.iterrows(), total=len(experiences)):
                prompt = eval(row['prompt']) # a list
                responses = eval(row['response'])
                tokens_generated = eval(row['tokens_generated'])
                # tokens_generated = [row['tokens_generated']] #legacy

                for response, n_tokens in zip(responses, tokens_generated):
                    if (n_tokens < THRESHOLD_FOR_LOOPS if collect_type == 'non_recurrent' else n_tokens >= THRESHOLD_FOR_LOOPS):
                        new_row = {'prompt': prompt, 'response': [response]} # both lists, for correct csv saving
                        # if there're no identical rows in the collection
                        found_identical = False
                        for _, present_row in all_samples.iterrows():
                            # eval if str
                            if type(present_row['prompt']) == str:
                                present_row['prompt'] = eval(present_row['prompt'])
                            if type(present_row['response']) == str:
                                present_row['response'] = eval(present_row['response'])
                            if present_row['prompt'] == new_row['prompt'] \
                                and present_row['response'] == new_row['response']:
                                    found_identical = True
                                    break
                        if not found_identical:
                            all_samples.loc[len(all_samples)] = new_row
            all_samples.to_csv(collection_path, index=False)

            print('number of ', collect_type, ' samples:', len(all_samples))

@click.command()
@click.option('-s', help='source folder containing experiment folders')
@click.option('-t', help='target folder to save the collected samples')
def main(s, t):
    collect_samples(s, t)

if __name__ == '__main__':
    main()