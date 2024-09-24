# append project root to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.llm.hookedLLM import HookedLLM
from src.llm.decoder_activation import LLMDecoderActivation
import torch

def get_feature(hooked_llm: HookedLLM, config: dict, response: str):
    # recover sequence
    sequence = response

    # get desired prefix
    detection_length = config['detection_length']
    whole_token_ids = hooked_llm.encode(sequence)[0]
    length = len(whole_token_ids)
    prefix_token_ids = whole_token_ids[:detection_length]
    desired_prefix = hooked_llm.decode(prefix_token_ids)

    # similarity matrix
    activation_prober = LLMDecoderActivation(hooked_llm)
    activation = activation_prober.decoder_activation(desired_prefix)
    # crop seq_len to detection_length
    activation = activation[:, :detection_length, :]
    similarity_matrix = activation_prober.similarity_rate_matrix(activation)

    # fill diagonal with 0
    similarity_matrix.fill_diagonal_(0)

    # find max for each token
    max_rate = similarity_matrix.max(dim=1).values

    # fill 0 to detection_length
    if len(max_rate) < detection_length:
        max_rate = torch.cat((max_rate, torch.zeros(detection_length - len(max_rate))), dim=0)

    # the final feature should be sorted max_rate concatenated with original max_rate
    sorted_max_rate, sorted_index = max_rate.sort(descending=True)
    return torch.cat((sorted_max_rate, max_rate[sorted_index]), dim=0), length