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

INPUT_FILE = '../data/gpqa_diamond.jsonl'
OUTPUT_FILE = '../data/responses/gpqa_diamond_mcq/gpqa_diamond_mcq_sampling_5_2.jsonl'
CUDA_VISIBLE_DEVICES = '6'
INDEX = 0

MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
TOKENIZER_NAME = MODEL_NAME
BATCH = 1                           # 1 will be used as a single inference. For batch inference, use a number greater than 1.
CONTEXT_LENGTH = 1024
TEMPERATURE = 0
CACHE_DIR = '/data/data/shri/Huggingface_model_cache'   # '/scratch/dhanda/huggingface_cache'
HUGGINGFACE_TOKEN = open('keys/huggingface.key').read().strip()

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
            max_new_tokens = 512,
            # do_sample = True
        )
        return outputs[0]["generated_text"][-1]['content']
    
    def extract_ideas(self, ideas):
        numbered_points = re.findall(r'\d+\.\s(.*?)(?=\n\n\d+\.|\Z)', ideas, re.DOTALL)
        return [point.strip() for point in numbered_points]

    def get_ideas(self, question, number_ideas):
        ######### GSM8k #########
        # text = f'You are an expert in mathematical reasoning. You will be given a mathematical problem. Please return {number_ideas} different ways you can solve this problem. Do NOT give a solution, just your a high-level idea of how you can solve this problem. Be as creative as possible, going beyond what you think is intuitively correct. Give your ideas in a numbered list starting from 1.\n\n[PROBLEM]\n{question}'
        ######### GPQA-Diamond, MCQ #########
        text = f'You are an expert in mathematics and science. You will be given a graduate-level scientific problem. Please return {number_ideas} different ways you can solve this problem. Do NOT give a solution, just your a high-level idea of how you can solve this problem. Be as creative as possible, going beyond what you think is intuitively correct. Give your ideas in a numbered list starting from 1.\n\n[PROBLEM]\n{question}'
        
        ideas_correct = False
        no_retries = 0
        while not ideas_correct and no_retries < 5:
            no_retries += 1
            ideas = self.get_response(text)
            extract_ideas = self.extract_ideas(ideas)
            if len(extract_ideas) == number_ideas:
                ideas_correct = True
        return ideas, extract_ideas

    def idea_sampling(self, input_file, output_file, number_ideas=1, number_responses_per_idea=1, index=0):
        data = self._load_data(input_file, index)
        with tqdm(total=len(data)) as pbar:
            for ele in data:
                ele['ideas'], ele['extracted_ideas'] = self.get_ideas(ele['question'], number_ideas)
                ele['responses'] = []
                for idea in ele['extracted_ideas']:
                    for _ in range(number_responses_per_idea):
                        ######### GSM8k #########
                        # ele['responses'].append(self.get_response(f'You are an expert in mathematical reasoning. You will be given a problem statement and an idea of how to solve that problem. Solve the question using that idea, and give the final answer in the following format\nThe final answer is: <YOUR FINAL ANSWER>. ONLY use the idea to solve that problem, do NOT use your own intuition.\n\n[PROBLEM]\n{ele["question"]}\n\n[IDEA]\n{idea}'))
                        ######### GPQA-Diamond #########
                        # ele['responses'].append(self.get_response(f'You are an expert in mathematics and science. You will be given a graduate-level problem and an idea of how to solve that problem. Solve the question using that idea, and give the final answer in the following format\nThe final answer is: <YOUR FINAL ANSWER>. ONLY use the idea to solve that problem, do NOT use your own intuition.\n\n[PROBLEM]\n{ele["question"]}\n\n[IDEA]\n{idea}'))
                        ######### GPQA-Diamond-MCQ #########
                        ele['responses'].append(self.get_response(f'You are an expert in mathematics and science. You will be given a graduate-level problem and an idea of how to solve that problem. Solve the question using that idea, and select the correct choice. ONLY use the idea to solve that problem, do NOT use your own intuition.\n\n[PROBLEM]\n{ele["question"]}\n\n[CHOICES]\n{"\n".join(f"{chr(97+i)}) {item.strip()}" for i, item in enumerate(ele["options"]))}\n\n[IDEA]\n{idea}\n\nGive the final answer in the following format\nThe final answer is: <FINAL ANSWER>. '))

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
    model.idea_sampling(args.input_file, args.output_file, number_ideas=5, number_responses_per_idea=2, index=args.index)


if __name__ == "__main__":
    main()