from pathlib import Path
import json
from tqdm import tqdm
import re

SAVE = True
# MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
# MODEL_NAME = 'iaa_fine_tuned_llama_model'
DATASET = 'math'

# FILE = f'{DATASET}_rs_10.json'
# FILE = f'{DATASET}_gs_20_5.json'
FILE = f'{DATASET}_tot_5_3_4.json'

if 'fine_tuned' in MODEL_NAME:
    FILE_PATH = f'../../data/sft_results/{MODEL_NAME}/{FILE}'
    VALIDATE_PATH = f'../../data/sft_results/{MODEL_NAME}/{FILE.replace(".json", "_validated.jsonl")}'
else:
    FILE_PATH = f'../../data/responses/{MODEL_NAME.split("/")[-1]}/{FILE}'
    if FILE.endswith('.json'):
        VALIDATE_PATH = f'../../data/validated/{MODEL_NAME.split("/")[-1]}/{FILE.replace(".json", ".jsonl")}'
    else:
        VALIDATE_PATH = f'../../data/validated/{MODEL_NAME.split("/")[-1]}/{FILE}'


def latex_to_expr(latex_str):
    """
    Used for Converting GroundTruth LaTeX-style fractions to a sympy-compatible string.
    E.g., "\\frac{14}{3}" → "14/3"
    """
    latex_str = latex_str.strip()
    # Replace LaTeX \frac{a}{b} with a/b
    latex_str = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', latex_str)
    return latex_str

def extract_answers(ele):
    if isinstance(ele['responses'][0], list):    # For GS
        responses = [resp for idea_responses in ele['responses'] for resp in idea_responses]
    else:  # For RS
        responses = ele['responses']
    answers = []
    for response in responses:
        try:
            final_answer = response.lower().split('**final answer**')[1].strip()
            final_answer = latex_to_expr(final_answer).strip().replace('(','').replace(')', '').replace(' ', '')
        except Exception as e:
            final_answer = ''
        answers.append(final_answer.strip())
    return answers

def is_instance_correct(ele):
    ele['correctness'] = []
    for answer in ele['extracted_answers']:
        ele['correctness'].append(latex_to_expr(ele['answer'].strip()).strip().replace('(','').replace(')', '').replace(' ', '') in answer)
    return True in ele['correctness']

if __name__ == '__main__':
    if FILE_PATH.endswith('.jsonl'):
        with open(FILE_PATH, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    elif FILE_PATH.endswith('.json'):
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
    Path(VALIDATE_PATH).parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    variants = []
    correct_per_ele = []

    with tqdm(total=len(data)) as pbar:
        for i, ele in enumerate(data):
            ele['extracted_answers'] = extract_answers(ele)
            ele['variants'] = len(set(ele['extracted_answers']))
            variants.append(len(set(ele['extracted_answers'])))

            if is_instance_correct(ele):
                ele['correct'] = True
                correct += 1
            else:
                ele['correct'] = False
            correct_per_ele.append(sum(ele['correctness'])/len(ele['correctness']))
            

            if SAVE:
                with open(VALIDATE_PATH, 'a+') as f:
                    f.write(json.dumps(ele)+'\n')
            accuracy = (correct / (i+1)) * 100
            pbar.set_description(f'Accuracy: {accuracy:.2f}%')
            pbar.update(1)
        
    print(f'For file {FILE}')
    print(f'Correct: {correct}. Total: {len(data)}. Accuracy: {correct/len(data)*100:.2f}%')
    
    if 'gs' in FILE:
        ideas = []
        for ele in data:
            ideas.append(len(ele['ideas']))
        print(f'Avg Ideas: {sum(ideas)/len(ideas):.2f}')