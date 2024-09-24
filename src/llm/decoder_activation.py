"""
This is used to load the probablities predicted by the model
"""

# append project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import click
import yaml
import torch

from accelerate import Accelerator
from transformers import (
    LlamaForCausalLM,
    LlamaModel,
)

from src.llm.hookedLLM import *
from typing import *

class LLMDecoderActivation():
    def __init__(self, llm: HookedLLM) -> None:
        self.llm = llm

    def decoder_activation(self, sequence: str) -> torch.Tensor:
        """
            output: boolean tensor of shape (num_layers, seq_len, intermediate_size), where each element is True if the corresponding neuron is activated
        """
        # Tokenize the input sequence
        inputs = self.llm.tokenize(sequence)
        
        # Dictionary to store activations
        activations = {}

        # generate a hook function that knows where to put the output obtained
        # to see if a 'neuron' is activated, we just have to see what is fed to the activation function
        def hook_generator(layer_idx: int):
            def hook_fn(module, input):
                (input,) = input # remove grad
                activations[layer_idx] = input
            return hook_fn

        # Register hooks to MLP layers
        hooks: Hooks = []
        # going down the structure of Llama to find the activation module of each decoder layer
        llamaForCausalLM: LlamaForCausalLM = self.llm.model
        llamaModel: LlamaModel = llamaForCausalLM.model
        decoder_layers: torch.nn.ModuleList = llamaModel.layers
        for layer_id, layer in enumerate(decoder_layers):
            mlp_of_the_layer = layer.mlp
            gate_proj_of_the_mlp = mlp_of_the_layer.act_fn
            # create the hook
            hooks.append((gate_proj_of_the_mlp, hook_generator(layer_id)))

        # Run the model to trigger hooks
        with torch.no_grad():
            with self.llm.hooks_applied(hooks):
                outputs = self.llm.model(**inputs)

        # integrate the results
        results = []
        for layer_id in range(len(decoder_layers)):
            results.append((activations[layer_id] > 0).to('cuda:0'))
        return torch.concat(results, dim=0)
    
    def similarity_rate(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
            input: two boolean tensors of same shape
            output: the ratio of entries that are the same
        """
        return (a == b).float().mean().item()
    
    CHUNK_SIZE = 6 # this suits: Llama7B, GPU Mem 48G, seq len 4k
    def similarity_rate_matrix(self, activations: torch.Tensor) -> torch.Tensor:
        """
            output: a tensor of shape (seq_len, seq_len) where each element is the similarity rate between the activations of the two positions
        """
        # change to (seq_len, num_layers * intermediate_size)
        seq_len = activations.shape[1]
        activations_squeeze = activations.permute(1, 0, 2).reshape(seq_len, -1)
        similarity_rate_matrix = torch.zeros(seq_len, seq_len)
        # fill in the similarity rates chunk by chunk
        num_chunks = (seq_len + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        for i in range(num_chunks):
            start_i = i * self.CHUNK_SIZE
            end_i = min((i + 1) * self.CHUNK_SIZE, seq_len)
            for j in range(num_chunks):
                start_j = j * self.CHUNK_SIZE
                end_j = min((j + 1) * self.CHUNK_SIZE, seq_len)
                activations_a = activations_squeeze[start_i:end_i].unsqueeze(1) # shape: (chunk_size, 1, num_layers * intermediate_size)
                activations_b = activations_squeeze[start_j:end_j].unsqueeze(0) # shape: (1, chunk_size, num_layers * intermediate_size)
                similarity_rate_matrix[start_i:end_i, start_j:end_j] = (activations_a == activations_b).float().mean(dim=-1)
        return similarity_rate_matrix
    
    def similarity_rate_matrix_endtoend(self, sequence: str) -> torch.Tensor:
        activations = self.decoder_activation(sequence)
        return self.similarity_rate_matrix(activations)
    
    LOOPING_SIMILARITY_THRESHOLD = 0.92
    def find_loop_start(
        self, 
        similarity_rate_matrix: torch.Tensor,
        similarity_threshold: float=LOOPING_SIMILARITY_THRESHOLD,
    ) -> Dict:
        """
            output: the start position of the loop, or None if no loop is found
        """
        def max_similarity_to_previous(location: int) -> int:
            """
                return: position, similarity rate    
            """
            position = similarity_rate_matrix[location, :location].argmax().item()
            similarity_rate = similarity_rate_matrix[location, position].item()
            return position, similarity_rate
        
        def last_similarity_to_previous_above_threshold(location: int) -> int:
            """
                return: position, None if no similarity above threshold
            """
            previous_similarities = similarity_rate_matrix[location, :location]
            above_threshold_mask = previous_similarities > similarity_threshold
            if above_threshold_mask.sum() == 0:
                return None
            position = above_threshold_mask.nonzero().max().item()
            return position

        # step 1: find a position highly linked to somewhere else at the end
        seq_len = similarity_rate_matrix.shape[0]
        repeating_point = None # not found
        for location in range(seq_len - 1, int(seq_len * 0.7), -1):
            position, similarity_rate = max_similarity_to_previous(location)
            print(f"location: {location}, position: {position}, similarity_rate: {similarity_rate}")
            if similarity_rate > similarity_threshold:
                repeating_point = location
                break
        if repeating_point is None:
            return None
        
        # step 2: going back the sequence along the similarity path
        trace = [repeating_point]
        while True:
            repeating_point = last_similarity_to_previous_above_threshold(repeating_point)
            if repeating_point is None:
                break
            trace.append(repeating_point)

        return {
            'loop_start': repeating_point,
            'trace': trace,
        }
    
    def layer_activation_rate(self, activations: torch.Tensor) -> torch.Tensor:
        """
            output: a tensor of shape (seq_len, num_layers,) where each element is the average activation rate of the layer
        """
        return activations.permute(1, 0, 2).float().mean(dim=-1)

@click.command()
@click.option('-exp_folder', help='Location of the experiment.')
def main(exp_folder):
    # Load the config file
    config_path = os.path.join(exp_folder, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    from src.llm.hookedLLM import HookedLLM
    hooked_llm = HookedLLM(**config['model'])

    prompt = "Hey there! How are you doing? Hello guys! How are you?"
    hooked_llm.print_tokens(prompt)
    # create activation hooker
    llm_decoder_activation = LLMDecoderActivation(hooked_llm)
    # Get the activations
    # activations = llm_decoder_activation.decoder_activation("How is the campaign")
    # print(activations)
    # print(activations.shape)
    similarity_rate_matrix = llm_decoder_activation.similarity_rate_matrix_endtoend(prompt)
    print(similarity_rate_matrix)

if __name__ == "__main__":
    main()