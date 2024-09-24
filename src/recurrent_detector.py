# append project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import pandas as pd
import yaml
import torch
from tqdm import tqdm
import click

from sklearn.model_selection import train_test_split

from src.mitigate.utils.mlp import MLPClassifier

@click.command()
@click.option('-exp_folder', help='experiment folder')
def main(exp_folder):
    # if not extracted, extract features
    if not os.path.exists(os.path.join(exp_folder, 'data.pt')):
        from src.mitigate.extract_feature import extract_feature_for_exp
        extract_feature_for_exp(exp_folder)
    # load features and labels
    data = torch.load(os.path.join(exp_folder, 'data.pt'))
    features = data['feature']
    labels = data['label']

    # load config
    with open(f'{exp_folder}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Perform the stratified split
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, train_size=config['mitigation']['train_ratio'], stratify=labels#, random_state=42
    )

    # make mlp classifier
    classifier = MLPClassifier(config['mitigation']['MLP_cfg'], train_features, train_labels)

    # save classify stats
    stats = classifier.evaluate(test_features, test_labels)

    stats.to_csv(os.path.join(exp_folder, 'classify_stats.csv'), index=False)

    print('test stats:')
    print(stats)

    # save train stats
    train_stats = classifier.evaluate(train_features, train_labels)
    train_stats.to_csv(os.path.join(exp_folder, 'train_stats.csv'), index=False)

if __name__ == "__main__":
    main()