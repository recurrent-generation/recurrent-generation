import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

import argparse
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
import yaml
import pandas as pd
import time

import numpy as np

from tqdm import tqdm
import transformers

from sklearn.metrics import confusion_matrix

class ExpData():
    """
    This is the dataset class for the features extracted from the hooked LLM, for experiment.
    Given the parent folder of the dataset, this class will load the data according to the preset structure.
    The structure of the data is specified as:
    ./features.npy
    ./labels.npy
    """
    def __init__(self, data_path: str):
        self.features = torch.tensor(np.load(f'{data_path}/features.npy'))
        self.labels = torch.tensor(np.load(f'{data_path}/labels.npy'))

class MLP(nn.Module):
    def __init__(self, config, input_size: int):
        super(MLP, self).__init__()
        layers = []
        for layer_config in config['model']['layers']:
            if layer_config['type'] == 'Linear':
                layers.append(nn.Linear(layer_config['input_size'], layer_config['output_size']))
            elif layer_config['type'] == 'ReLU':
                layers.append(nn.ReLU())
            elif layer_config['type'] == 'Sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_config['type'] == 'LayerNorm':
                layers.append(nn.LayerNorm(layer_config['normalized_shape']))
            # Add more layer types as needed
        # add implicit first linear layer
        layers = [nn.Linear(input_size, config['model']['layers'][0]['input_size']), nn.ReLU()] + layers
        # initialize weights
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
class MLPClassifier:
    def __init__(self, config, train_features: torch.Tensor, train_labels: torch.Tensor) -> None:
        self.config = config
        self.train_features = train_features
        # Convert labels to float
        self.train_labels = train_labels.float()

        # Normalize the training data
        self.mean = train_features.mean(dim=0)
        self.std = train_features.std(dim=0)
        self.train_features = (train_features - self.mean) / self.std

        # calculate class weights
        class_counts = train_labels.long().bincount()
        self.class_weights = 1 / class_counts.float()
        self.class_weights /= self.class_weights.sum()
        if 'class_weights' in config['training']:
            self.class_weights *= torch.tensor(config['training']['class_weights'])
            # normalize
            self.class_weights /= self.class_weights.sum()
        print(f'Class weights: {self.class_weights}')

        self.classify_threshold = 0.5 if 'classify_threshold' not in config['training'] else config['training']['classify_threshold']

        self.input_size = train_features.shape[1]
        self.model = MLP(config, self.input_size)
        self.train_model()

    def train_model(self):
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"Training on {device}")

        use_scheduler = True if 'use_scheduler' in self.config['training'] and self.config['training']['use_scheduler'] else False
        print(f"Using scheduler: {use_scheduler}")

        # Move the model to the appropriate device
        self.model.to(device)

        self.model.train()

        batch_size = self.config['training']['batch_size']
        num_epochs = self.config['training']['epochs']
        learning_rate = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        dataset = TensorDataset(self.train_features.to(device), self.train_labels.to(device))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if use_scheduler:
            num_steps = len(data_loader) * num_epochs
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_steps)

        criterion = nn.BCELoss(reduction='none')

        for epoch_id in tqdm(range(num_epochs), desc="Epochs"):
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)  # Outputs are raw logits
                # Apply class weights manually
                loss = criterion(outputs, labels[..., None])
                weights = labels * self.class_weights[1] + (1 - labels) * self.class_weights[0]
                loss = (loss * weights).mean()  # Manually weight the loss and take the mean
                loss.backward()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
            # print(f'Epoch {epoch_id+1}/{num_epochs}, Loss: {loss.item()}')

    def predict(self, features: torch.Tensor):
        """
        features: (N, num_features)
        """
        features = features.to(self.device)
        
        # Determine the device of the features tensor
        device = features.device

        # Move the mean and std tensors to the same device as features
        mean = self.mean.to(device)
        std = self.std.to(device)

        # Normalize the features
        features = (features - mean) / std

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # predict per batch
            predictions = []
            batch_size = self.config['training']['batch_size']
            for i in range(0, features.shape[0], batch_size):
                outputs = self.model(features[i:i+batch_size])
                # Assuming binary classification with a threshold of 0.5
                predictions.append((outputs > self.classify_threshold).float())

            predictions = torch.cat(predictions, dim=0)

        return predictions

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> pd.DataFrame:
        """
        features: (N, num_features)
        labels: (N,)
        """
        start_time = time.time()
        predictions = self.predict(features)
        average_time_cost = (time.time() - start_time) / features.shape[0] * self.config['training']['batch_size']
        print(f'Average prediction time: {average_time_cost}')
        
        tp, fn, fp, tn = confusion_matrix(
            labels.cpu().numpy(), 
            predictions.cpu().numpy(), 
            labels=[1, 0] # positive, negative
        ).ravel()

        fpr = fp / (fp + tn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)

        # Save statistics to CSV
        stats = pd.DataFrame({
            'Metric': ['Recall', 'False Positive Rate', 'Precision', 'F1 Score', 'Accuracy'],
            'Value': [recall, fpr, precision, f1, accuracy]
        })

        return stats

    def score(self, features: torch.Tensor):
        features = features.to(self.device)
        
        # Determine the device of the features tensor
        device = features.device

        # Move the mean and std tensors to the same device as features
        mean = self.mean.to(device)
        std = self.std.to(device)

        # Normalize the features
        features = (features - mean) / std

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(features)

        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample data based on configuration in experiment folder.')
    parser.add_argument('experiment_folder', type=str, help='Path to the experiment folder.')

    args = parser.parse_args()

    experiment_folder = args.experiment_folder

    import yaml
    with open(f'{experiment_folder}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # load train data
    train_data = ExpData(config['training']['data_path'])

    classifier = MLPClassifier(config, train_data.features, train_data.labels)

    # eval
    test_data = ExpData(config['eval']['data_path'])
    accuracy = classifier.evaluate(test_data.features, test_data.labels)
    print(f'Accuracy: {accuracy}')