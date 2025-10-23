import os
from pathlib import Path
import click
import json

import torch.distributed as dist
from huggingface_hub import login
from transformers import AutoTokenizer
import vllm


MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'
TOKENIZER_NAME = MODEL_NAME
MODEL_FOR_EVALUATION = 'Llama-3.2-3B-Instruct'
DATASET = 'olympiad_bench'
INPUT_FILE = f'../data/validated/{MODEL_FOR_EVALUATION}/{DATASET}_rs_100.jsonl'
OUTPUT_FILE = INPUT_FILE.replace('.jsonl', '_concepts.jsonl').replace('validated', 'concepts')
TEST_MODE = False
INDEX = 0

CUDA_VISIBLE_DEVICES = '3'
CONTEXT_LENGTH = 1024
TEMPERATURE = 0.8
CACHE_DIR = '/data/data/dhanda/huggingface_cache'
HUGGINGFACE_TOKEN = open('keys/huggingface.key', 'r').read().strip()
os.environ['HF_HOME'] = CACHE_DIR

class ConceptExtractor():
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
            gpu_memory_utilization = 0.60,            # The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher values will increase the KV cache size and thus improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors.
        )
    
    def cleanup(self):
        self.llm = None
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def _load_data(self, file_path, index):
        with open(file_path, 'r') as f:
            data = [json.loads(ele) for ele in f.readlines()][index:]
        return data[:3] if self.test_mode else data

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
            elif self.model_name.split('/')[0] == 'Qwen':
                formatted_prompts.append(self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            else:
                raise ValueError(f"Unsupported model name's format: {self.model_name}")
        return formatted_prompts

    def get_responses(self, prompts, no_responses=1):
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
        for output in outputs:
            response_ele = []
            for ele in output.outputs:
                response_ele.append(ele.text)
            if no_responses == 1:
                responses.append(response_ele[0])
            else:
                responses.append(response_ele)
        
        return responses

    def concept_prompt(self, ele, response):
        return f'''
You are ConceptTagger, an expert that maps a worked-out solution (chain-of-thought or final answer) to the most specific mathematical or logical concept that makes the solution possible.

Task: For every input consisting of a reasoning explanation (a step-by-step solution, scratch-work, or short justification):
1. Read the explanation.
2. Decide which single mathematical concept, theorem, or canonical formula is essential for the solution.
3. Output that concept’s standard name—nothing else.

Choose the narrowest concept that still covers the whole solution.
• Good: "Pythagorean Theorem" (precise).
• Bad: "Geometry" (too broad).
If two or more concepts appear, pick the one without which the problem cannot be solved (typically the first pivotal step).

Here are two examples:

### Example 1
Problem: A right triangle has legs of lengths 5 cm and 12 cm. What is the length of the hypotenuse?
Step-by-step solution:
Step 1: Recognize this is a right triangle → apply the Pythagorean Theorem.
Step 2: hypotenuse = $\\sqrt{(5^2 + 12^2)} = \\sqrt{(25 + 144)} = \\sqrt{169} = 13 cm$
Concept Used: Pythagorean Theorem

### Example 2
Problem: What is the area of a rectangle with a length of 9 meters and width of 4 meters?
Step-by-step solution:
Step 1: Identify the shape as a rectangle.
Step 2: Use the area formula: Area = length × width = 9 × 4 = 36 m²
Concept Used: Area of Rectangle

Formatting Rules:
Output exactly one line with the concept name.
Use Title Case and the singular form (e.g., "Least Common Multiple", not "LCMs").
No extra punctuation, explanation, or line breaks.

Extract the concept from the following:
Problem: {ele['question']}
Step-by-step solution:
{response}
Concept Used:
        '''.strip()

    def extract_concepts(self, input_file, output_file, index=0):
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            data = self._load_data(input_file, index)

            for ele in data:
                prompts = []
                for response in ele['responses']:
                    prompts.append(self.concept_prompt(ele, response))

                ele['concepts'] = self.get_responses(prompts, no_responses=1)
            self._save_data(output_file, data)
        finally:
            self.cleanup()
    
@click.command()
@click.option('--model', default=MODEL_NAME, help='Model name or path. Compatible with Hugging Face models.')
@click.option('--tokenizer', default=TOKENIZER_NAME, help='Tokenizer name or path. Compatible with Hugging Face models.')
@click.option('--input-file', default=INPUT_FILE, help='Input File that is processed.')
@click.option('--output-file', default=OUTPUT_FILE, help='Output File where results are stored.')
@click.option('--test-mode', type=bool, default=TEST_MODE, help='Test mode.')
@click.option('--index', type=int, default=INDEX, help='Starting index for evaluation.')
@click.option('--cuda-visible-devices', default=CUDA_VISIBLE_DEVICES, help='CUDA_VISIBLE_DEVICES')
@click.option('--context-length', type=int, default=CONTEXT_LENGTH, help='Maximum length of the context.')
@click.option('--temperature', type=float, default=TEMPERATURE, help='Temperature for sampling.')
@click.option('--cache-dir', default=CACHE_DIR, help='Directory to cache models.')
@click.option('--huggingface-token', default=HUGGINGFACE_TOKEN, help='Hugging Face API token.')
def main(model, tokenizer, input_file, output_file, test_mode, index, cuda_visible_devices, context_length, temperature, cache_dir, huggingface_token):
    concept_extractor = ConceptExtractor(
        cuda_visible_devices=cuda_visible_devices,
        model_name=model,
        tokenizer_name=tokenizer,
        context_length=context_length,
        temperature=temperature,
        cache_dir=cache_dir,
        huggingface_token=huggingface_token,
        test_mode=test_mode,
    )
    concept_extractor.extract_concepts(
        input_file=input_file,
        output_file=output_file,
        index=index
    )

if __name__ == '__main__':
    main()