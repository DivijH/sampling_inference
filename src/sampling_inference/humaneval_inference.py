import os
from vllm_inference import VLLMInferenceBase
import click



MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
# MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
# MODEL_NAME = 'google/gemma-3-27b-it'
# MODEL_NAME = '../../trained_models/iaa_fine_tuned_llama_model/checkpoint-100'
TOKENIZER_NAME = MODEL_NAME
DATASET = 'humaneval'
INPUT_FILE = f'../../data/{DATASET}.jsonl'
CUDA_VISIBLE_DEVICES = '2'
INDEX = 0
FINISH_REASON = True

#### For RS
NUMBER_SAMPLES = 1
OUTPUT_FILE = f'../../data/responses/{MODEL_NAME.split("/")[-1]}/{DATASET}_rs_1.json'
# OUTPUT_FILE = f'../../data/sft_results/{MODEL_NAME.split("/")[-2]}/{DATASET}_rs_10.json'

#### For GS
MAX_IDEAS = 5
NUMBER_RESPONSES_PER_IDEA = 20
# OUTPUT_FILE = f'../../data/responses/{MODEL_NAME.split("/")[-1]}/{DATASET}_gs_{MAX_IDEAS}_{NUMBER_RESPONSES_PER_IDEA}.json'


CONTEXT_LENGTH = 4096
TEMPERATURE = 0.8
CACHE_DIR = '/data/data/dhanda/huggingface_cache'
HUGGINGFACE_TOKEN = open('../keys/huggingface.key', 'r').read().strip()
os.environ['HF_HOME'] = CACHE_DIR
TEST_MODE = True  # Run with 3 samples for testing, with just 3 samples for each question


class HumanevalVLLMInference(VLLMInferenceBase):
    def rs_prompt(self, ele):
        if self.model_name.split('/')[0] in ['meta-llama', 'Qwen', 'google']:
            return f'''
You are an expert python programmer. Your task is to complete the following function.

Follow these instructions precisely:
1.  Before writing code, present your solution as a step-by-step process.
2.  Explain each step clearly and concisely.
3.  Do not write test-cases or docstrings.
4.  Wrap the code in ```python ``` block.
5.  Use correct Python syntax throughout your solution.

QUESTION:
{ele['question']}
'''.strip()
        elif 'trained_models' in self.model_name:
            return ele['question'] + '\nWrap the code in ```python ``` block.'
        else:
            raise ValueError(f"Unsupported model name's format: {self.model_name}")
    
    def gs_initial_prompt(self, ele):
        return f'''
You are an expert python programmer. You will be presented with a programming question and your task is to identify and state one single, specific concept that is most relevant and useful for solving the problem.

QUESTION:
{ele['question']}

Provide only the name or short description of the concept, that is most directly applicable to solving this problem. Do not attempt to solve the original question. Only provide the concept.
'''.strip()
    
    def gs_more_prompt(self, ele, ideas_text):
        return f'''
You are an expert python programmer. You will be presented with a programming question and a list of concepts that have already been proposed as potentially useful for solving the question. Your task is to provide a *new* and *different* concept that is most relevant and useful for solving the question.

QUESTION:
{ele['question']}

EXISTING CONCEPTS:
{ideas_text}

Provide only the name or the short description of the concept, that is most directly applicable to solving this problem. Do not attempt to solve the original question. Only provide the concept. If no new, distinct, and useful concept can be identified, respond with "No additional concepts found."
'''
    
    def gs_prompt(self, ele, idea):
        return f'''
You are an expert python programmer. Your task is to complete the following function only using the given concept.

Follow these instructions precisely:
1.  Before writing code, present your solution as a step-by-step process.
2.  Explain each step clearly and concisely.
3.  Do not write test-cases or docstrings.
4.  Wrap the code in ```python ``` block.
5.  Use correct Python syntax throughout your solution.
6.  Only use the provided concept to solve the problem. Do not use any other concepts.

QUESTION:
{ele['question']}

CONCEPT:
{idea}
'''.strip()

@click.command()
@click.option('--cuda-visible-devices', default=CUDA_VISIBLE_DEVICES, help='CUDA_VISIBLE_DEVICES')
@click.option('--model', default=MODEL_NAME, help='Model name or path. Compatible with Hugging Face models.')
@click.option('--tokenizer', default=TOKENIZER_NAME, help='Tokenizer name or path. Compatible with Hugging Face models.')
@click.option('--context-length', type=int, default=CONTEXT_LENGTH, help='Maximum length of the context.')
@click.option('--temperature', type=float, default=TEMPERATURE, help='Temperature for sampling.')
@click.option('--index', type=int, default=INDEX, help='Starting index for evaluation.')
@click.option('--finish-reason', type=bool, default=FINISH_REASON, help='Finish reason.')
@click.option('--cache-dir', default=CACHE_DIR, help='Directory to cache models.')
@click.option('--huggingface-token', default=HUGGINGFACE_TOKEN, help='Hugging Face API token.')
@click.option('--input-file', default=INPUT_FILE, help='Input File that is processed.')
@click.option('--output-file', default=OUTPUT_FILE, help='Output File where results are stored.')
@click.option('--number-samples', type=int, default=NUMBER_SAMPLES, help='Number of samples for RS.')
@click.option('--number-responses-per-idea', type=int, default=NUMBER_RESPONSES_PER_IDEA, help='Number of responses per idea.')
@click.option('--max-ideas', type=int, default=MAX_IDEAS, help='Number of ideas.')
@click.option('--test-mode', type=bool, default=TEST_MODE, help='Test mode.')
def main(cuda_visible_devices, model, tokenizer, context_length, temperature, index, finish_reason, cache_dir, huggingface_token, input_file, output_file, number_samples, number_responses_per_idea, max_ideas, test_mode):
    model = HumanevalVLLMInference(
        cuda_visible_devices = cuda_visible_devices,
        model_name = model,
        tokenizer_name = tokenizer,
        context_length = context_length,
        temperature = temperature,
        cache_dir = cache_dir,
        huggingface_token = huggingface_token,
        test_mode = test_mode
    )
    if 'rs' in output_file:
        model.rs(
            input_file = input_file,
            output_file = output_file,
            number_responses = number_samples,
            index = index,
            get_finish_reason = finish_reason
        )
    elif 'gs' in output_file:
        model.gs(
            input_file = input_file,
            output_file = output_file,
            max_ideas = max_ideas,
            number_responses_per_idea = number_responses_per_idea,
            index = index
        )
    else:
        raise ValueError(f"Unsupported Inference Algorithm in output file: {output_file}")
    model.cleanup()
    print("Done")

if __name__ == '__main__':
    main()