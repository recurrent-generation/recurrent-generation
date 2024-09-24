from contextlib import contextmanager
from typing import List, Tuple, Callable, Optional, Union

import torch as t
from torch import nn
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    LlamaForCausalLM
)

import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import gc
import yaml

from accelerate import Accelerator

# look at the comment of nn.Module.register_forward_pre_hook
# the function applies a pre_hook before the forward pass of the module
# each time before the module receives its input, the hook will be called,
# having the module ref and the input tensor as arguments
# if it returns a tensor, the input tensor will be replaced by the returned tensor
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
"""
    Args: 
        module: the module that will receive the input tensor
        input: the input tensor
    Returns:
        Optional[t.Tensor]: modified input tensor if not None
"""

Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]

class HookedLLM:
    def __init__(
        self,
        path: str,
        sampling_params: dict = None,
        injections: dict = None
    ) -> None:
        self.path = path
        self.accelerator: Accelerator = Accelerator()
        # load
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)
        self.model.eval()
        self.model.tie_weights()
        if self.tokenizer.eos_token_id is None:
            eos_token = self.tokenizer.eos_token
            self.tokenizer.eos_token_id = self.tokenizer.encode(eos_token)[0]
            print(f"EOS token id is {self.tokenizer.eos_token_id}")

        if 'gemma' in path:
            # special: for gemma, eos is eot, id is 107
            self.tokenizer.eos_token_id = 107
        # wrap tokenizer
        def tokenize(text: str|List[str], pad_length: Optional[int] = None) -> BatchEncoding:
            """Tokenize prompts onto the appropriate devices."""

            if pad_length is None:
                padding_status = 'longest'
            else:
                padding_status = "max_length"
            with t.no_grad():
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=padding_status,
                    add_special_tokens=False
                    # max_length=pad_length,
                )

                for key in tokens:
                    tokens[key] = tokens[key].to('cuda')
                return self.accelerator.prepare(tokens)
        self.tokenize = tokenize
        # set sampling params
        self.set_generation_config(sampling_params)
        # apply injections
        if injections is not None:
            for _, inject_params in injections.items():
                if not inject_params['status'] == "active":
                    continue
                if inject_params['type'] == 'DifferenceSteering':
                    hook = self.get_difference_steering_hook(
                        **inject_params['parameters']
                    )
                    self.apply_hooks([hook])
                    print(f"Hook applied with the following configurations: {inject_params['parameters']}")
                else:
                    raise ValueError(f"Unsupported injection type: {inject_params['type']}.")

    def get_blocks(self):
        """Get the blocks of the model."""
        if isinstance(self.model, PreTrainedModel):
            try:
                # from transformers.models.llama.modeling_llama import LlamaForCausalLM
                return self.model.model.layers
            except:
                # from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
                return self.model.transformer.h
        raise ValueError(f"Unsupported model type: {type(self.model)}.")
    
    def get_num_layers(self):
        return len(self.get_blocks())

    def apply_hooks(self, hooks: Hooks):
        """Apply hooks to the model."""
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        return handles

    @contextmanager
    def hooks_applied(self, hooks: Hooks):
        """Should be called as `with hooks_applied(hooks):` to apply the hooks in the context. The hooks will be removed after the context ends."""
        handles = []
        try:
            handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
            yield # context prepared
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def padding_token_changed(self, padding_str):
        """Temporarily change the torch tokenizer padding token."""
        # Preserve original padding token state.
        original_padding_token_id = self.tokenizer.pad_token_id

        if padding_str in self.tokenizer.get_vocab():
            padding_id = self.tokenizer.convert_tokens_to_ids(padding_str)
        else:
            raise ValueError("Padding string is not in the tokenizer vocabulary.")

        # Change padding token state.
        self.tokenizer.pad_token_id = padding_id

        # Context manager boilerplate.
        try:
            yield
        finally:
            # Revert padding token state.
            self.tokenizer.pad_token_id = original_padding_token_id

    def get_layer_input(self, layers: list[int], input: str, pad_length: Optional[int] = None) -> list[t.Tensor]:
        """
            Args:
                layers: the layers of which the inputs are wanted
                input: the prompt
            
            Returns:
                a list of tensors, the desired input to the layers. Retval[i] is the input to the i'th layer, None if not required in LAYERS. Shape of the tensor: (1, n_token, n_dimension)
        """
        modded_streams = [None] * len(self.get_blocks())
        # Factory function that builds the initial hooks.
        def obtain_input(layerId):
            def _helper_hook(module, current_inputs):
                modded_streams[layerId] = current_inputs[0]
            return _helper_hook
        hooks = [
            (layer, obtain_input(i))
            for i, layer in enumerate(self.get_blocks()) if i in layers
        ]
        # with self.padding_token_changed(PAD_TOKEN):
        model_input = self.tokenize(input, pad_length=pad_length)
        with t.no_grad():
            # Register the hooks.
            with self.hooks_applied(hooks):
                self.model(**model_input)
        return modded_streams

    def get_difference_steering_hook(
        self,
        minus_prompt: str = "The following is a", # this seems to have no meaning, thus can work like subtracting the mean to get the deviation
        plus_prompt: str = "help the bad guy",
        inject_layer_idx: int = 10,
        coefficient = 4.0
    ) -> Hook:
        """
            Args:
                minus_prompt: the prompt for the negative steering vector
                plus_prompt: the prompt for the positive steering vector
                inject_layer_idx: the layer to inject the steering vector into

            Returns:
                steering_vector: HookedLLM.SteeringVector, the steering vector
        """
        len_minus, len_plus = (len(self.tokenizer.encode(y)) for y in [minus_prompt, plus_prompt])
        if len_minus != len_plus:
            print("Warning: The lengths of the minus and plus prompts are different:" + f"{len_minus} vs {len_plus}")
        steering_vec_len = max(len_minus, len_plus)
        plus_vec = self.get_layer_input(
            [inject_layer_idx], plus_prompt, pad_length=steering_vec_len
        )[inject_layer_idx]
        minus_vec = self.get_layer_input(
            [inject_layer_idx], minus_prompt, pad_length=steering_vec_len
        )[inject_layer_idx]
        steering_addition = (plus_vec - minus_vec) * coefficient
        print(steering_addition.shape)

        def steering_hook(_, current_inputs: Tuple[t.Tensor]) -> None:
            (resid_pre,) = current_inputs
            # Only add to the first forward-pass, not to later tokens.
            if resid_pre.shape[1] == 1:
                # Caching in `model.generate` for new tokens.
                return
            if resid_pre.shape[1] > steering_addition.shape[1]:
                resid_pre[:, :steering_addition.shape[1]] += steering_addition
            else:
                resid_pre += steering_addition[:, :resid_pre.shape[1]]
                print("Warning: Steering addition has more tokens than resid_pre.")

        return (self.get_blocks()[inject_layer_idx], steering_hook)

    def forward(self, input: str, pad_length: Optional[int] = None):
        """
            Args:
                input: the prompt
                pad_length: the length to pad the input to

            Returns:
                output: the output of the model, shape: (1, n_token, n_dimension)
        """
        # with self.padding_token_changed(PAD_TOKEN):
        model_input = self.tokenize(input, pad_length=pad_length)
        return self.model(**model_input)
    
    def set_generation_config(self, configs):
        """
            Set the generation configuration for the model
            Args:
                **configs: the configuration parameters for the model's generation
        """
        self.generation_configs = configs

    def prepare_for_chat(self, prompt: List[str]) -> BatchEncoding:
        """
            Prepare the input for the chat
        """
        return self.tokenize(
            [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{p}"}],
                    tokenize=False
                )
                for p in prompt
            ]
        )
    
    def generate_wrapper(self, input_ids: t.Tensor, attention_mask: t.Tensor, generation_config: GenerationConfig) -> t.Tensor:
        # if 'gemma-2' in self.path:
        if False:
            from transformers import HybridCache

            # # Prepare a cache class and pass it to model's forward
            # # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
            max_generated_length = input_ids.shape[1] + 10
            past_key_values = HybridCache(config=self.model.config, batch_size=1, max_cache_len=max_generated_length, device=self.model.device, dtype=self.model.dtype)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True)
            past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
            return outputs
        else:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
    
    def chat(
        self, prompt: Union[str, List[str]]
    ) -> List[str]:
        if isinstance(prompt, str):
            prompt = [prompt]

        input_tokenization = self.prepare_for_chat(prompt)

        assert hasattr(self, 'generation_configs'), "The instance must have 'generation_configs' attribute"

        # get a temporary generation config with num_return_sequences set to num_samples
        temp_config_dict = self.generation_configs.copy()
        generation_config = GenerationConfig(
            **temp_config_dict,
            pad_token_id = self.tokenizer.pad_token_id
        )
        
        # input_tokenization: BatchEncoding = self.tokenize(prompt)
        with t.no_grad():
            tokens: t.Tensor = self.generate_wrapper(
                input_ids=input_tokenization.input_ids,
                attention_mask=input_tokenization.attention_mask,
                generation_config=generation_config
            )
    
        response = []
        for i, output_encoding in enumerate(tokens):
            input_token_length = len(input_tokenization.encodings[i].ids)
            # find the first eos token
            eos_token_id = self.tokenizer.eos_token_id
            response_encoding = output_encoding[input_token_length:]
            eos_token_idx = (
                (response_encoding == eos_token_id) | (response_encoding == self.tokenizer.pad_token_id)
            ).nonzero(as_tuple=True)[0]
            if len(eos_token_idx) > 0:
                eos_token_idx = eos_token_idx[0]
                response.append(
                    self.tokenizer.decode(
                        response_encoding[:eos_token_idx],
                    )
                )
            else:
                # print decoded output
                # print('error: no EOS token found in the output')
                # print('full sequence:')
                # print(self.tokenizer.decode(output_encoding))
                # question_encoding = output_encoding[:input_token_length]
                # print('question:')
                # print(self.tokenizer.decode(question_encoding))
                # print('it will be treated as an empty string response')
                # response.append('')
                # raise ValueError("No EOS token found in the output.")

                # the reason is that instead of reaching a eos/eot, the model reaches the maximum length
                # thus we need to decode the whole sequence
                response.append(self.tokenizer.decode(response_encoding))

        return response # prompt removed, just the response
    
    def recover_sequence(self, prompt: str, response: str) -> str:
        """
            During the processing of the model generated content, we only have the prompt and the generated answer. This function recovers the whole sequence the model is looking at.
        """
        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]
        chat_template_applied = self.tokenizer.decode(self.tokenizer.apply_chat_template(messages, tokenize=True))
        return chat_template_applied + response
    
    def get_num_tokens(self, prompt: str) -> int:
        """
            Get the number of tokens in the prompt
        """
        return len(self.tokenizer.encode(prompt))
    
    def destroy(self):
        """
        Destroy the model and free the GPU memory
        """
        print("destroying model...")
        self.model = None
        self.tokenizer = None

        # Manually trigger Python's garbage collector
        gc.collect()
        with t.no_grad():
            self.accelerator.clear()
            t.cuda.empty_cache()

    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def encode(self, text: str) -> t.Tensor:
        """
            Encode one str to token ids, batch size is automatically 1 at dim 0
        """
        return self.tokenizer.encode(text, return_tensors="pt").to('cuda')
    
    def print_tokens(self, text: str):
        """
            Print the tokens of the text
        """
        print('the text:', text)
        print("tokens of the text:")
        token_ids = self.encode(text)
        for id in token_ids[0]:
            print(self.decode(id))

    def token_locations(self, text: str) -> List[Tuple[int, int]]:
        """
            Get the token locations of the text
        """

        # Comprehensive mapping of special tokens to their corresponding characters
        token_to_char = {
            '<0x0A>': '\n',  # Line break
            '<0x20>': ' ',   # Space
            '<0x09>': '\t',  # Tab
            '<0x0D>': '\r',  # Carriage return
            # Add other special tokens and their corresponding characters here if needed
        }

        tokens = [token_to_char[original_token] if original_token in token_to_char else original_token 
                  for original_token in self.tokenizer.tokenize(text)]

        token_locations = []
        start = 0
        for token in tokens:
            pure_token = token.replace('▁', '')
            start = text.find(pure_token, start)
            end = start + len(pure_token)
            token_locations.append((start, end))
            start = end
        return token_locations

    def decode(self, tokens: Union[t.Tensor, np.ndarray]) -> str:
        """
            Decode tokens to text
        """
        return self.tokenizer.decode(tokens)
    
    def get_probabilities(self, input: t.Tensor) -> t.Tensor:
        """
            Get the probabilities of the tokens
        """
        assert input.shape[0] == 1, "Only one input is allowed"
        with t.no_grad():
            outputs = self.model(input)
            logits = outputs.logits
            return t.softmax(logits[0, -1, :], dim=-1)
        
    def get_token(self, id: int) -> str:
        """
            Get the token from the id
        """
        return self.tokenizer.decode(id)
    
    def next_token_deterministic(self, input: t.Tensor) -> Tuple[int, float]:
        """
            Get the next token deterministically
        """
        probs = self.get_probabilities(input)
        id = t.argmax(probs)
        return id, probs[id].item()
    
    def get_eos_token(self) -> int:
        """
            Get the end of sequence token
        """
        return self.tokenizer.eos_token_id