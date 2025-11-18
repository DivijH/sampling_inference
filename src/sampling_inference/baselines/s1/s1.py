import os
from pathlib import Path
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from abc import abstractmethod

import torch.distributed as dist
import vllm
from transformers import AutoTokenizer
from huggingface_hub import login


# Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
MAX_TOKENS_THINKING = 32000


class s1InferenceBase:
    def __init__(
        self,
        cuda_visible_devices: str,        
        model_name: str,
        tokenizer_name: str,
        context_length: int,
        temperature: float,
        cache_dir: str,
        huggingface_token: str,
        max_tokens_thinking: int,
        max_iterations: int,
        test_mode: bool = False,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.context_length = context_length
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.test_mode = test_mode
        self.max_tokens_thinking = max_tokens_thinking
        self.max_iterations = max_iterations

        login(token=huggingface_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=cache_dir)
        self.llm = vllm.LLM(
            model = self.model_name,
            tokenizer = self.tokenizer_name,
            task = 'generate',
            enforce_eager = False,
            download_dir = self.cache_dir,
            trust_remote_code = True,
            gpu_memory_utilization = 0.60,
        )

    def cleanup(self):
        self.llm = None
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def _load_data(self, input_file: str, index: int):
        if input_file.endswith('.jsonl'):
            with open(input_file, 'r') as f:
                data = [json.loads(line) for line in f]
        elif input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {input_file}")
        return data[:3] if self.test_mode else data[index:]

    def _save_data(self, file_path: str, data: list):
        file_ext = Path(file_path).suffix.lower()
        with open(file_path, 'w') as f:
            if file_ext == '.json':
                json.dump(data, f, indent=4)
            elif file_ext == '.jsonl':
                for ele in data:
                    f.write(json.dumps(ele) + '\n')

    def _format_prompt(self, ele, is_initial):
        if self.model_name.split('/')[0] == 'meta-llama' or 'trained_models' in self.model_name:
            # self.stop_token_ids = self.tokenizer('<|eot_id|>')['input_ids']
            prompt = '<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n' + ele['question'] + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nthink'
        elif self.model_name.split('/')[0] == 'Qwen':
            prompt = '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + ele['question'] + '<|im_end|>\n<|im_start|>assistant\n'
            if not is_initial:
                prompt += ' Wait,'.join(ele['responses']) + ' Wait,'
        else:
            raise ValueError(f"Unsupported model name's format: {self.model_name}")
        return prompt
    
    def get_response(self, ele, no_responses=1, is_initial=True, is_final=False):
        if self.test_mode:
            no_responses = min(no_responses, 3)
        
        if not is_final:
            prompt = self._format_prompt(ele, is_initial)
        else:
            prompt = self.final_prompt(ele)

        sampling_params = vllm.SamplingParams(
            n = no_responses,                   # Number of output sequences to return for the given prompt.
            temperature = self.temperature,     # Float that controls the randomness of the sampling. Zero means greedy sampling.
            top_p = 0.5,                        # Float that controls the cumulative probability of the top tokens to consider. Set to 1 to consider all tokens.
            top_k = -1,                         # Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.
            max_tokens = self.context_length,   # Maximum number of tokens to generate per output sequence.
            skip_special_tokens = True,         # Whether to skip special tokens in the output.
        )
        outputs = self.llm.generate(
            prompts = [prompt],
            sampling_params = sampling_params,
            use_tqdm = True
        )
        print('-'*50)
        print(prompt)
        print(outputs[0].outputs[0].text)
        print('*'*25)
        print('*'*50)
        
        return outputs[0].outputs[0].text

    def s1_instance(self, ele):
        # First LLM call
        ele['responses'] = [self.get_response(ele, no_responses=1, is_initial=True, is_final=False)]

        # Subsequent LLM calls
        counter = 1
        remaining_tokens = self.max_tokens_thinking - len(self.tokenizer(ele['responses'][0])['input_ids'])
        while remaining_tokens > 0 and counter < self.max_iterations:
            continued_response = self.get_response(ele, no_responses=1, is_initial=False, is_final=False)
            ele['responses'].append(continued_response)
            remaining_tokens -= len(self.tokenizer(continued_response)['input_ids'])
            counter += 1
        
        # Final LLM call
        # ele['final_response'] = self.get_response(ele, no_responses=1, is_initial=False, is_final=True)

    def s1(self, input_file, output_file, index=0):
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            data = self._load_data(input_file, index)

            for ele in data:
                ele['responses'] = self.s1_instance(ele)

            self._save_data(output_file, data)
        finally:
            self.cleanup()

    @abstractmethod
    def final_prompt(self, ele):
        pass

if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = "1"
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    CONTEXT_LENGTH = 32768
    TEMPERATURE = 0.0
    CACHE_DIR = "/data/data/dhanda/huggingface_cache"
    HUGGINFACE_TOKEN = open('../../../keys/huggingface.key', 'r').read().strip()
    MAX_TOKENS_THINKING = 32000
    MAX_ITERATIONS = 2
    TEST_MODE = True

    s1 = s1InferenceBase(
        cuda_visible_devices=CUDA_VISIBLE_DEVICES,
        model_name=MODEL_NAME,
        tokenizer_name=MODEL_NAME,
        context_length=CONTEXT_LENGTH,
        temperature=TEMPERATURE,
        cache_dir=CACHE_DIR,
        huggingface_token=HUGGINFACE_TOKEN,
        max_tokens_thinking=MAX_TOKENS_THINKING,
        max_iterations=MAX_ITERATIONS,
        test_mode=TEST_MODE,
    )
    ele = {
        "question": "What is the meaning of life?",
    }
    s1.s1_instance(ele)
    print(' Wait,'.join(ele['responses']))