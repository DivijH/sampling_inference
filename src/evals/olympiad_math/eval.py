import os
from pathlib import Path
import json
from pprint import pprint
from tqdm import tqdm
from math_judger import MathJudger

SAVE = False
DATASET = 'olympiad_bench'
# FILE = f'{DATASET}_inference.jsonl'
FILE = f'{DATASET}_bon_10.jsonl'
# FILE = f'{DATASET}_sampling_5_2.jsonl'
# FILE = f'{DATASET}_sampling_3_3.jsonl'
FILE_PATH = f'../../../data/responses/{DATASET}/{FILE}'
VALIDATE_PATH = f'../../../data/validated/{DATASET}/{FILE}'

def extract_answers(ele):
    responses = ele['responses']
    answers = []
    for response in responses:
        final_answer = response.split('final answer is')[-1].strip()
        answers.append(final_answer)
    return answers

def is_value_correct(ele, answer, judger):
    if ele['answer_type'] and 'Tuple' in ele['answer_type']:
        judge_result = judger.judge(answer, ele['final_answer'][0])
    else:
        if ele['error']:
            if ',' in ele['error']:
                precisions = ele['error'].split(',')
                precisions = [float(p) if p else 1e-8 for p in precisions]
                judge_result = judger.judge(answer, ele['final_answer'][0] if ele.get('final_answer') else '', precisions)
            else:
                precision = float(ele['error'])
                judge_result = judger.judge(answer, ele['final_answer'][0] if ele.get('final_answer') else '', precision)
        else:
            judge_result = judger.judge(answer, ele['final_answer'][0] if ele.get('final_answer') else '')
    return judge_result

def is_instance_correct(ele, judger):
    ele['correctness'] = []
    for answer in ele['extracted_answers']:
        ele['correctness'].append(is_value_correct(ele, answer, judger))
    return True in ele['correctness']


if __name__ == '__main__':
    with open(FILE_PATH, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    Path(VALIDATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    correct = 0
    variants = []

    math_correct = 0
    math_total = 0
    physics_correct = 0
    physics_total = 0
    judger = MathJudger()

    # answer_types = []
    # for ele in data:
    #     answer_types.append(ele['answer_type'])
    # print(set(answer_types)) # {'Equation', 'Interval', 'Tuple', 'Equation,Numerical', 'Numerical', 'Expression', 'Expression,Numerical', 'Numerical,Expression'}
    # exit()

    with tqdm(total=len(data)) as pbar:
        for ele in data:
            # if 'Equation'.lower() in ele['answer_type'].lower():
            ele['extracted_answers'] = extract_answers(ele)
            ele['variants'] = len(set(ele['extracted_answers']))
            variants.append(len(set(ele['extracted_answers'])))

            if is_instance_correct(ele, judger):
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
                    f.write(json.dumps(ele)+'\n')
            pbar.update(1)
        
    print(f'For file {FILE}')
    print(f'Correct: {correct}. Total: {len(data)}. Accuracy: {correct/len(data)*100:.2f}%')
    print(f'Math Correct: {math_correct}. Math Total: {math_total}. Math Accuracy: {math_correct/math_total*100:.2f}%')
    print(f'Physics Correct: {physics_correct}. Physics Total: {physics_total}. Physics Accuracy: {physics_correct/physics_total*100:.2f}%')
    print(f'Avg Exploration: {sum(variants)/len(variants):.2f}')
