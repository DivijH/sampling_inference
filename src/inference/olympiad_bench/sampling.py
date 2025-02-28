'''
Remember to update the idea prompt and the question prompt.
'''

import os
from pathlib import Path
import argparse
import json
import re
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

os.environ['HF_HOME'] = '/data/data/shri/Huggingface_model_cache'

INPUT_FILE = '../../../data/olympiad_bench.jsonl'
OUTPUT_FILE = '../../../data/responses/olympiad_bench/olympiad_bench_sampling_3_3.jsonl'
CUDA_VISIBLE_DEVICES = '1'
NUMBER_IDEAS = 3
NUMBER_RESPONSES_PER_IDEA = 3
INDEX = 0

MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
TOKENIZER_NAME = MODEL_NAME
BATCH = 1                           # 1 will be used as a single inference. For batch inference, use a number greater than 1.
CONTEXT_LENGTH = 1024
TEMPERATURE = 0
CACHE_DIR = '/data/data/shri/Huggingface_model_cache'   # '/scratch/dhanda/huggingface_cache'
HUGGINGFACE_TOKEN = open('../../keys/huggingface.key').read().strip()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_visible_devices', type=str, default=CUDA_VISIBLE_DEVICES, required=False, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--model', type=str, default=MODEL_NAME, required=False, help='Model name or path. Compatible with Hugging Face models.')
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_NAME, required=False, help='Tokenizer name or path. Compatible with Hugging Face models.')
    parser.add_argument('--batch', type=int, default=BATCH, required=False, help='Number for batch inference where several prompts are sent to the model at once.')
    parser.add_argument('--context_length', type=int, default=CONTEXT_LENGTH, required=False, help='Maximum length of the context.')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, required=False, help='Temperature for sampling.')
    parser.add_argument('--index', type=int, default=INDEX, required=False, help='Starting index for evaluation.')
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR, required=False, help='Directory to cache models.')
    parser.add_argument('--huggingface_token', type=str, default=HUGGINGFACE_TOKEN, required=False, help='Hugging Face API token.')
    parser.add_argument('--input_file', type=str, default=INPUT_FILE, required=False, help='Input File that is processed.')
    parser.add_argument('--output_file', type=str, default=OUTPUT_FILE, required=False, help='Output File where results are stored.')
    parser.add_argument('--number_ideas', type=int, default=NUMBER_IDEAS, required=False, help='Number of ideas to be generated.')
    parser.add_argument('--number_responses_per_idea', type=int, default=NUMBER_RESPONSES_PER_IDEA, required=False, help='Number of responses to be generated per idea.')
    return parser.parse_args()

class Model:
    def __init__(
            self,
            cuda_visible_devices: str,
            model_name: str,
            tokenizer_name: str,
            context_length: int,
            temperature: float,
            cache_dir: str,
            huggingface_token: str
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.context_length = context_length
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.huggingface_token = huggingface_token
        self.pipe = pipeline(
            "text-generation",
            model = self.model_name,
            token = self.huggingface_token,
        )
    
    def _load_data(self, file_path, index):
        with open(file_path, 'r') as f:
            data = [json.loads(ele) for ele in f.readlines()][index:]
        return data

    def _save_data(self, file_path, ele):
        with open(file_path, 'a+') as f:
            f.write(json.dumps(ele)+'\n')
    
    def get_response(self, text):
        outputs = self.pipe([
                # {"role": "system", "content": "You answer the question in one word"},
                {"role": "user", "content": text},
            ],
            max_new_tokens = self.context_length,
            # do_sample = True
        )
        return outputs[0]["generated_text"][-1]['content']
    
    def extract_ideas(self, ideas):
        numbered_points = re.findall(r'\d+\.\s(.*?)(?=\n\n\d+\.|\Z)', ideas, re.DOTALL)
        return [point.strip() for point in numbered_points]

    def get_ideas(self, ele, number_ideas):
        text = f'''
You are an expert in {ele['subject']}. Read the given question and return {number_ideas} different ways you can solve this question.

[QUESTION]
{ele['question']}

[CONSTRAINTS]
1. Do NOT give a solution, just {number_ideas} ways of how you can solve this question.
2. Be as creative as possible, going beyond what you think is intuitively correct.
3. Do NOT use coding or simulations to solve this question. Instead, provide methods or theories that can be used to solve the question.

[FORMAT]
Return {number_ideas} ways in the following format:

1. <IDEA 1>
2. <IDEA 2>
...
{number_ideas}. <IDEA {number_ideas}>
'''
        
        ideas_correct = False
        no_retries = 0
        while not ideas_correct and no_retries < 5:
            no_retries += 1
            ideas = self.get_response(text.strip())
            extract_ideas = self.extract_ideas(ideas)
            if len(extract_ideas) == number_ideas:
                ideas_correct = True
        return ideas, extract_ideas

    def idea_sampling(self, input_file, output_file, number_ideas=1, number_responses_per_idea=1, index=0):
        data = self._load_data(input_file, index)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with tqdm(total=len(data)) as pbar:
            for ele in data:
                ele['ideas'], ele['extracted_ideas'] = self.get_ideas(ele, number_ideas)
                ele['responses'] = []
                for idea in ele['extracted_ideas']:
                    for _ in range(number_responses_per_idea):
                        ele['responses'].append(self.get_response(f'''
You are an expert in {ele['subject']}. Solve the given question using the provided idea.

[PROBLEM]
{ele['question']}

[IDEA]
{idea}

[CONSTRAINTS]
Only use the idea to solve the question. Do NOT use your own intuition.

[FORMAT]
First give the reasoning of how to solve the problem. Then state your final answer as "The final answer is <FINAL ANSWER>".
'''.strip()))
                self._save_data(output_file, ele)
                pbar.update(1)


def main():
    args = get_args()

    model = Model(
        cuda_visible_devices = args.cuda_visible_devices,
        model_name = args.model,
        tokenizer_name = args.tokenizer,
        context_length = args.context_length,
        temperature = args.temperature,
        cache_dir = args.cache_dir,
        huggingface_token = args.huggingface_token
    )
    model.idea_sampling(
        input_file = args.input_file,
        output_file = args.output_file,
        number_ideas = args.number_ideas,
        number_responses_per_idea = args.number_responses_per_idea,
        index = args.index
    )


if __name__ == "__main__":
    main()