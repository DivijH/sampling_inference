from pathlib import Path
import json
from tqdm import tqdm
import sys
sys.set_int_max_str_digits(1_000_000)

from math_verify import parse
from math_verify.parser import (
    LatexExtractionConfig,
    ExprExtractionConfig
)


SAVE = True
# MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
# MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
MODEL_NAME = 'google/gemma-3-27b-it'
# MODEL_NAME = 'iaa_fine_tuned_llama_model'
DATASET = 'olympiad_bench'

# FILE = f'{DATASET}_rs_100.json'
# FILE = f'{DATASET}_gs_5_20.json'
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

def extract_answers(ele):
    def parse_ans(text):
        if isinstance(text, list):
            parsed_answer = parse(text[0], extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])
            parsed_answer.append(text[0])
        else:
            parsed_answer = parse(text, extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])
            parsed_answer.append(text)
        return parsed_answer
    
    ele['parsed_golden_answer'] = parse_ans(ele['final_answer'])
    if isinstance(ele['responses'][0], list):  # For GS
        responses = [resp for idea_responses in ele['responses'] for resp in idea_responses]
    else:  # For RS
        responses = ele['responses']
    answers = []
    for response in responses:
        response = response.strip().lower()
        if "final answer" not in response:
            answers.append([])
        else:
            final_answer = parse_ans(response.split('final answer')[-1].strip())
            answers.append(final_answer)
    return answers

def is_instance_correct(ele):
    ele['correctness'] = []
    for answer in ele['extracted_answers']:
        try:
            ele['correctness'].append(bool(set(ele['parsed_golden_answer']) & set(answer)))
        except TypeError as e:
            # Handle unhashable types by converting to strings
            golden_answer_strs = set(str(item) for item in ele['parsed_golden_answer'])
            answer_strs = set(str(item) for item in answer)
            ele['correctness'].append(bool(golden_answer_strs & answer_strs))
    return True in ele['correctness']


if __name__ == '__main__':
    with open(FILE_PATH, 'r') as f:
        data = json.load(f)
    Path(VALIDATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    correct = 0
    no_ideas = 0

    math_correct = 0
    math_total = 0
    physics_correct = 0
    physics_total = 0

    with tqdm(total=len(data)) as pbar:
        for ele in data:
            ele['extracted_answers'] = extract_answers(ele)
            
            if 'no_ideas' in ele:
                no_ideas += ele['no_ideas']

            if is_instance_correct(ele):
                if ele['subject'] == 'Physics':
                    physics_correct += 1
                    physics_total += 1
                else:
                    math_correct += 1
                    math_total += 1

                ele['correct'] = True
                correct += 1
            else:
                if ele['subject'] == 'Physics':
                    physics_total += 1
                else:
                    math_total += 1

                ele['correct'] = False

            if SAVE:
                with open(VALIDATE_PATH, 'a+') as f:
                    f.write(json.dumps(ele, default=str)+'\n')
            pbar.update(1)
        
    print(f'For file {FILE}')
    print(f'Correct: {correct}. Total: {len(data)}. Accuracy: {correct/len(data)*100:.2f}%')
    if math_total > 0:
        print(f'Math Correct: {math_correct}. Math Total: {math_total}. Math Accuracy: {math_correct/math_total*100:.2f}%')
    if physics_total > 0:
        print(f'Physics Correct: {physics_correct}. Physics Total: {physics_total}. Physics Accuracy: {physics_correct/physics_total*100:.2f}%')
    if no_ideas > 0:
        print(f'No Avg Ideas: {no_ideas/len(data):.2f}')
