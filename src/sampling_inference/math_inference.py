import os
from vllm_inference import VLLMInferenceBase
import click



# MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
# MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
MODEL_NAME = '../../trained_models/iaa_fine_tuned_llama_model/checkpoint-100'
TOKENIZER_NAME = MODEL_NAME
DATASET = 'math'
INPUT_FILE = f'../../data/{DATASET}.jsonl'
CUDA_VISIBLE_DEVICES = '4'
INDEX = 0

#### For RS
NUMBER_SAMPLES = 10
# OUTPUT_FILE = f'../../data/responses/{MODEL_NAME.split("/")[-1]}/{DATASET}_rs_100.json'
OUTPUT_FILE = f'../../data/sft_results/{MODEL_NAME.split("/")[-2]}/{DATASET}_rs_10.json'

#### For GS
MAX_IDEAS = 5
NUMBER_RESPONSES_PER_IDEA = 20
# OUTPUT_FILE = f'../../data/responses/{MODEL_NAME.split("/")[-1]}/{DATASET}_gs_{MAX_IDEAS}_{NUMBER_RESPONSES_PER_IDEA}.json'


CONTEXT_LENGTH = 1024
TEMPERATURE = 0.8
CACHE_DIR = '/data/data/dhanda/huggingface_cache'
HUGGINGFACE_TOKEN = open('../keys/huggingface.key', 'r').read().strip()
os.environ['HF_HOME'] = CACHE_DIR
TEST_MODE = False  # Run with 3 samples for testing, with just 3 samples for each question


class MathVLLMInference(VLLMInferenceBase):    
    def rs_prompt(self, ele):
        if self.model_name.split('/')[0] in ['meta-llama', 'Qwen']:
            return f'''
You are an expert mathematician. Your task is to first list all relevant theorems and then use one of them to answer the question in a step-by-step manner.

Follow these instructions precisely:
1.  First, identify relevant theorems from the question.
2.  Then, present your solution as a step-by-step process, showing all calculations and reasoning, using one of the identified theorems.
3.  Explain each step clearly and concisely.
4.  Use correct mathematical notation throughout your solution.
5.  Conclude with the heading "**Final Answer**" followed by the answer. Just write the final answer here, without any additional text.

QUESTION:
{ele['question']}

**Theorems**
<theorems>

**Step-by-Step Solution**
<using theorem i ...>

**Final Answer**
<final answer here>
'''.strip()
        elif 'trained_models' in self.model_name:
            return ele['question']
        else:
            raise ValueError(f"Unsupported model name's format: {self.model_name}")
    
    def gs_initial_prompt(self, ele):
        return f'''
You are an expert mathematician. You will be presented with a mathematical question and your task is to identify and state one single, specific theorem or fundamental concept that is most relevant and useful for solving the problem.

QUESTION:
{ele['question']}

Provide only the name of the theorem or concept, or a concise statement of the principle, that is most directly applicable to solving this problem. Do not attempt to solve the original problem. Only provide the theorem or concept.
'''.strip()
    
    def gs_more_prompt(self, ele, ideas_text):
        return f'''
You are an expert mathematician. You will be presented with a mathematical question and a list of theorems and concepts that have already been proposed as potentially useful for solving the problem. Your task is to provide a *new* and *different* theorem or concept that is most relevant and useful for solving the problem.

QUESTION:
{ele['question']}

EXISTING CONCEPTS:
{ideas_text}

Provide only the name of the theorem or concept, or a concise statement of the principle, that is most directly applicable to solving this problem. Do not attempt to solve the original problem. Only provide the theorem or concept. If no new, distinct, and useful theorem or concept can be identified, respond with "No additional  concepts found."
'''
    
    def gs_prompt(self, ele, idea):
        return f'''
You are an expert mathematician. Your task is to answer mathematical questions with step-by-step solution using the given concept. In the end, you will provide the final answer under the heading **Final Answer**.

Follow these instructions precisely:
1.  Present your solution as a step-by-step process, then select the correct option.
2.  Explain each step clearly and concisely.
3.  Use correct mathematical notation throughout your solution.
4.  Solve the problem only using the given concept.
5.  In the end, write "**Final Answer**" followed by the answer.

QUESTION:
{ele['question']}

CONCEPT:
{idea}

**Step-by-Step Solution**
'''.strip()

@click.command()
@click.option('--cuda-visible-devices', default=CUDA_VISIBLE_DEVICES, help='CUDA_VISIBLE_DEVICES')
@click.option('--model', default=MODEL_NAME, help='Model name or path. Compatible with Hugging Face models.')
@click.option('--tokenizer', default=TOKENIZER_NAME, help='Tokenizer name or path. Compatible with Hugging Face models.')
@click.option('--context-length', type=int, default=CONTEXT_LENGTH, help='Maximum length of the context.')
@click.option('--temperature', type=float, default=TEMPERATURE, help='Temperature for sampling.')
@click.option('--index', type=int, default=INDEX, help='Starting index for evaluation.')
@click.option('--cache-dir', default=CACHE_DIR, help='Directory to cache models.')
@click.option('--huggingface-token', default=HUGGINGFACE_TOKEN, help='Hugging Face API token.')
@click.option('--input-file', default=INPUT_FILE, help='Input File that is processed.')
@click.option('--output-file', default=OUTPUT_FILE, help='Output File where results are stored.')
@click.option('--number-samples', type=int, default=NUMBER_SAMPLES, help='Number of samples for RS.')
@click.option('--number-responses-per-idea', type=int, default=NUMBER_RESPONSES_PER_IDEA, help='Number of responses per idea.')
@click.option('--max-ideas', type=int, default=MAX_IDEAS, help='Number of ideas.')
@click.option('--test-mode', type=bool, default=TEST_MODE, help='Test mode.')
def main(cuda_visible_devices, model, tokenizer, context_length, temperature, index, cache_dir, huggingface_token, input_file, output_file, number_samples, number_responses_per_idea, max_ideas, test_mode):
    model = MathVLLMInference(
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
            index = index
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