import os
from tot import ToTInferenceBase
import click


# MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
# MODEL_NAME = '../../../trained_models/iaa_fine_tuned_llama_model/checkpoint-100'
TOKENIZER_NAME = MODEL_NAME
DATASET = 'humaneval'
INPUT_FILE = f'../../../../data/{DATASET}.jsonl'
CUDA_VISIBLE_DEVICES = '3'
INDEX = 0

#### For ToT
# number_of_calls = (1 + number_thoughts) * (1 + (number_steps - 1) * top_k) + number_final_solutions
#                                            ^^^ total_paths                   ^^^ final_call
NUMBER_THOUGHTS = 5  # Number of thoughts to generate at each step
NUMBER_STEPS = 3     # Depth of the tree (number of reasoning steps)
TOP_K = 4            # Number of top thoughts to keep at each step
NUMBER_FINAL_SOLUTIONS = 100  # Number of complete solutions to generate from best paths
MODEL_SLUG = MODEL_NAME.split("/")[-1]
OUTPUT_FILE = f'../../../../data/responses/{MODEL_SLUG}/{DATASET}_tot_{NUMBER_THOUGHTS}_{NUMBER_STEPS}_{TOP_K}.json'


CONTEXT_LENGTH = 1024
TEMPERATURE = 0.8
CACHE_DIR = '/data/data/dhanda/huggingface_cache'
HUGGINGFACE_TOKEN = open('../../../keys/huggingface.key', 'r').read().strip()
os.environ['HF_HOME'] = CACHE_DIR
TEST_MODE = False  # Run with 3 samples for testing


class HumanevalToTInference(ToTInferenceBase):
    def thought_generation_prompt(self, ele, previous_thoughts=""):
        if previous_thoughts:
            context = (
                f"\n\nPREVIOUS REASONING STEPS:\n{previous_thoughts}\n\n"
                "Now, provide the NEXT reasoning step (≤2 sentences) to continue solving this problem."
            )
        else:
            context = (
                "\n\nProvide the FIRST reasoning step (≤2 sentences) to start solving this problem."
            )
        
        return f'''
You are an expert python programmer working on solving a programming question step-by-step.

QUESTION:
{ele['question']}
{context}

Provide ONE specific step that moves toward the solution. This should be:
- A single, focused insight or code snippet
- Based on programming principles or concepts
- A concrete step (not a vague statement)

Do not solve the entire problem - just provide the next step.

REASONING STEP:
'''.strip()
    

    def thought_evaluation_prompt(self, ele, thought_path):
        return f'''
You are an expert python programmer evaluating reasoning paths for solving programming questions.

QUESTION:
{ele['question']}

REASONING PATH:
{thought_path}

Evaluate how promising this reasoning path is for solving the problem. Consider:
1. Programming correctness of the steps
2. Relevance to the question
3. Logical progression toward a solution
4. Whether it's on the right track

Provide a score from 1 to 10, where:
- 1-3: Poor reasoning, incorrect approach, or irrelevant
- 4-6: Partially correct but incomplete or inefficient
- 7-9: Good reasoning, correct approach, likely to lead to solution
- 10: Excellent reasoning, very likely to solve the problem

Respond with ONLY a single number between 1 and 10.

SCORE:
'''.strip()
    
    def final_solution_prompt(self, ele, thought_path):
        return f'''
You are an expert python programmer. You have developed a reasoning path for solving a programming question. Now complete the solution with all necessary code and provide the final answer.

QUESTION:
{ele['question']}

REASONING PATH:
{thought_path}

Follow these instructions:
1. Continue from the reasoning path above
2. Complete all necessary code
3. Use correct Python syntax
4. Do not write test-cases or docstrings.
5. Wrap the code in ```python ``` block.
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
@click.option('--number-thoughts', type=int, default=NUMBER_THOUGHTS, help='Number of thoughts to generate at each step.')
@click.option('--number-steps', type=int, default=NUMBER_STEPS, help='Depth of the tree (number of reasoning steps).')
@click.option('--top-k', type=int, default=TOP_K, help='Number of top thoughts to keep at each step.')
@click.option('--number-final-solutions', type=int, default=NUMBER_FINAL_SOLUTIONS, help='Number of final solutions to generate.')
@click.option('--test-mode', type=bool, default=TEST_MODE, help='Test mode.')
def main(cuda_visible_devices, model, tokenizer, context_length, temperature, index, cache_dir, huggingface_token, 
         input_file, output_file, number_thoughts, number_steps, top_k, number_final_solutions, test_mode):
    model = HumanevalToTInference(
        cuda_visible_devices=cuda_visible_devices,
        model_name=model,
        tokenizer_name=tokenizer,
        context_length=context_length,
        temperature=temperature,
        cache_dir=cache_dir,
        huggingface_token=huggingface_token,
        test_mode=test_mode
    )
    
    model.tot(
        input_file=input_file,
        output_file=output_file,
        number_thoughts=number_thoughts,
        number_steps=number_steps,
        top_k=top_k,
        number_final_solutions=number_final_solutions,
        index=index
    )
    
    print("Done")


if __name__ == '__main__':
    main()

