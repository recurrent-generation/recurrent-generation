# append project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import os
import pandas as pd
import yaml
import torch
from tqdm import tqdm
import click

from src.mitigate.utils.feature import get_feature
from src.llm.hookedLLM import HookedLLM

# EXP_FOLDER = 'experiment/mitigation/gemma2-2b'
NUM_SAMPLES = 500 # for both recurrent and non-recurrent

LABELS = {
    'non_recurrent': 0,
    'recurrent': 1
}

def extract_feature_for_exp(exp_folder):
    NON_RECURRENT_COLLECTION_PATH = os.path.join(exp_folder, 'non_recurrent_samples.csv')
    RECURRENT_SAMPLE_PATH = os.path.join(exp_folder, 'recurrent_samples.csv')
    COLLECTION_PATHS = {
        'non_recurrent': NON_RECURRENT_COLLECTION_PATH,
        'recurrent': RECURRENT_SAMPLE_PATH
    }

    # load config
    with open(f'{exp_folder}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # prepare model
    hooked_llm = HookedLLM(**config['model'])

    data = {
        'feature': [],
        'label': [],
        'len': [],
    }
    for label in ['non_recurrent', 'recurrent']:
        collection_path = COLLECTION_PATHS[label]
        collection = pd.read_csv(collection_path)

        # sample data if bigger than NUM_SAMPLES
        if len(collection) > NUM_SAMPLES:
            sample = collection.sample(NUM_SAMPLES)
        else:
            sample = collection

        # get feature
        for idx, row in tqdm(sample.iterrows(), total=len(sample)):
            feature, length = get_feature(hooked_llm, config['mitigation'], row['response'])
            # feature_list.append(feature)
            # label_list.append(LABELS[label])
            data['feature'].append(feature)
            data['label'].append(LABELS[label])
            data['len'].append(length)

        # save features
        # feature_tensor = torch.stack(feature_list)
        # label_tensor = torch.tensor(label_list)
        # torch.save(feature_tensor, os.path.join(exp_folder, 'features.pt'))
        # torch.save(label_tensor, os.path.join(exp_folder, 'labels.pt'))
    data['feature'] = torch.stack(data['feature'])
    data['label'] = torch.tensor(data['label'])
    data['len'] = torch.tensor(data['len'])
    torch.save(data, os.path.join(exp_folder, 'data.pt'))

@click.command()
@click.option('-exp_folder', help='experiment folder')
def main(exp_folder):
    extract_feature_for_exp(exp_folder)

if __name__ == '__main__':
    main()