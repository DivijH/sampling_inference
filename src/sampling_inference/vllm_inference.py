import os
from pathlib import Path
import json
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch.distributed as dist
from huggingface_hub import login
from transformers import AutoTokenizer
import vllm


class VLLMInferenceBase(ABC):
    def __init__(
            self,
            cuda_visible_devices: str,
            model_name: str,
            tokenizer_name: str,
            context_length: int,
            temperature: float,
            cache_dir: str,
            huggingface_token: str,
            test_mode: bool = False,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=cache_dir)
        self.context_length = context_length
        self.temperature = temperature
        self.cache_dir = cache_dir
        login(token=huggingface_token)
        self.test_mode = test_mode
        self.llm = vllm.LLM(
            model = self.model_name,
            tokenizer = self.tokenizer_name,
            task = 'generate',
            enforce_eager = False,                      # Since, enforce-eager is enabled, async output processor cannot be used.
            download_dir = self.cache_dir,
            trust_remote_code = True,
            # swap_space: int = 4  # GiB                # The size (GiB) of CPU memory per GPU to use as swap space. This can be used for temporarily storing the states of the requests when their `best_of` sampling parameters are larger than 1. If all requests will have `best_of=1`, you can safely set this to 0. Otherwise, too small values may cause out-of-memory (OOM) errors.
            # tensor_parallel_size = 4,                 # The number of GPUs to use for distributed execution with tensor parallelism.
            # gpu_memory_utilization = 0.90,            # The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher values will increase the KV cache size and thus improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors.
        )
    
    def cleanup(self):
        self.llm = None
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def _load_data(self, file_path, index):
        with open(file_path, 'r') as f:
            data = [json.loads(ele) for ele in f.readlines()]
        return data[:3] if self.test_mode else data[index:]

    def _save_data(self, file_path, data):
        file_ext = Path(file_path).suffix.lower()
        with open(file_path, 'w') as f:
            if file_ext == '.json':
                json.dump(data, f, indent=4)
            elif file_ext == '.jsonl':
                for ele in data:
                    f.write(json.dumps(ele) + '\n')
    
    def _format_prompts(self, prompts):
        formatted_prompts = []
        for prompt in prompts:
            message = [{
                "role": "user",
                "content": prompt
            }]
            if self.model_name.split('/')[0] == 'meta-llama' or 'trained_models' in self.model_name:
                formatted_prompts.append('<|start_header_id|>user<|end_header_id|>\n\n' + prompt + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
            elif self.model_name.split('/')[0] in ['Qwen', 'google']:
                formatted_prompts.append(self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            else:
                raise ValueError(f"Unsupported model name's format: {self.model_name}")
        return formatted_prompts

    def get_responses(self, prompts, no_responses=1, get_finish_reason=False):
        if self.test_mode:
            no_responses = min(no_responses, 3)
        prompts = self._format_prompts(prompts)
        sampling_params = vllm.SamplingParams(
            n = no_responses,                   # Number of output sequences to return for the given prompt.
            temperature = self.temperature,     # Float that controls the randomness of the sampling. Zero means greedy sampling.
            top_p = 0.5,                        # Float that controls the cumulative probability of the top tokens to consider. Set to 1 to consider all tokens.
            top_k = -1,                         # Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
            max_tokens = self.context_length,   # Maximum number of tokens to generate per output sequence.
            skip_special_tokens = True,         # Whether to skip special tokens in the output.
        )
        outputs = self.llm.generate(
            prompts = prompts,
            sampling_params = sampling_params,
            use_tqdm = True
        )
        responses = []
        finish_reasons = []
        for output in outputs:
            response_ele = []
            finish_reason_ele = []
            for ele in output.outputs:
                response_ele.append(ele.text)
                finish_reason_ele.append(ele.finish_reason)
            responses.append(response_ele)
            finish_reasons.append(finish_reason_ele)
        if get_finish_reason:
            return responses, finish_reasons
        else:
            return responses

    @abstractmethod
    def rs_prompt(self, ele):
        pass

    def rs(self, input_file, output_file, number_responses=1, index=0, get_finish_reason=False):
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            data = self._load_data(input_file, index)

            prompts = []
            for ele in data:
                prompts.append(self.rs_prompt(ele))

            if get_finish_reason:
                responses, finish_reasons = self.get_responses(prompts, no_responses=number_responses, get_finish_reason=True)
                for response, finish_reason, data_ele in zip(responses, finish_reasons, data):
                    data_ele['responses'] = response
                    data_ele['finish_reason'] = finish_reason
            else:
                responses = self.get_responses(prompts, no_responses=number_responses)
                for response, data_ele in zip(responses, data):
                    data_ele['responses'] = response
            self._save_data(output_file, data)
        finally:
            self.cleanup()
    
    @abstractmethod
    def gs_initial_prompt(self, ele):
        pass
    
    @abstractmethod
    def gs_more_prompt(self, ele, ideas_text):
        pass
    
    @abstractmethod
    def gs_prompt(self, ele, idea):
        pass
    
    def get_ideas(self, data, max_ideas):
        '''This will generate all ideas for all the instances and return a list of list of ideas'''
        ideas_all = []
        for ele in tqdm(data, desc='Generating Ideas...'):
            current_ideas = []
            ideas_left = True
            iteration = 0
            while ideas_left and iteration < max_ideas:
                iteration += 1
                if not current_ideas:
                    prompt = self.gs_initial_prompt(ele)
                else:
                    ideas_text = '\n\n'.join([f'Idea {i+1}:\n{idea}' for i, idea in enumerate(current_ideas)])
                    prompt = self.gs_more_prompt(ele, ideas_text)
                response = self.get_responses([prompt], no_responses=1)[0][0]
                if "no more ideas" in response.lower() or "don't have any other" in response.lower() or "i don't have" in response.lower() or "no additional" in response.lower():
                    ideas_left = False
                else:
                    cleaned_response = response.strip()
                    current_ideas.append(cleaned_response)
                    current_ideas = list(set(current_ideas))
            ideas_all.append(current_ideas)
        return ideas_all
    
    def gs(self, input_file, output_file, max_ideas=10, number_responses_per_idea=1, index=0):
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            data = self._load_data(input_file, index)
            ideas_all = self.get_ideas(data, max_ideas)

            prompts = []
            for ideas, data_ele in zip(ideas_all, data):
                data_ele['no_ideas'] = len(ideas)
                data_ele['ideas'] = ideas
                for idea in ideas:
                    prompts.append(self.gs_prompt(data_ele, idea))
            
            responses = self.get_responses(prompts, no_responses=number_responses_per_idea)
            final_responses = []
            counter = 0
            for ideas in ideas_all:
                final_responses.append(responses[counter:counter+len(ideas)])
                counter += len(ideas)
            
            for final_response, data_ele in zip(final_responses, data):
                data_ele['responses'] = final_response
            self._save_data(output_file, data)
        finally:
            self.cleanup()


if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES = '7'
    MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
    TOKENIZER_NAME = MODEL_NAME
    CONTEXT_LENGTH = 1024
    TEMPERATURE = 0.8
    CACHE_DIR = '/data/data/dhanda/huggingface_cache'
    HUGGINGFACE_TOKEN = open('../keys/huggingface.key', 'r').read().strip()

    class TestVLLMInference(VLLMInferenceBase):
        def rs_prompt(self, ele):
            pass
        def gs_initial_prompt(self, ele):
            pass
        def gs_more_prompt(self, ele, ideas_text):
            pass
        def gs_prompt(self, ele, idea):
            pass
    
    vllm_inference = TestVLLMInference(
        cuda_visible_devices=CUDA_VISIBLE_DEVICES,
        model_name=MODEL_NAME,
        tokenizer_name=TOKENIZER_NAME,
        context_length=CONTEXT_LENGTH,
        temperature=TEMPERATURE,
        cache_dir=CACHE_DIR,
        huggingface_token=HUGGINGFACE_TOKEN
    )
    vllm_inference.get_responses(['What is the capital of France?'], no_responses=1)
    vllm_inference.cleanup()