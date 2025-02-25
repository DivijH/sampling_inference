import os
import json
from pprint import pprint
from tqdm import tqdm

DATASET = 'gpqa_diamond_mcq'
# FILE = f'{DATASET}_inference.jsonl'
# FILE = f'{DATASET}_bon_10.jsonl'
FILE = f'{DATASET}_sampling_5_2.jsonl'
FILE_PATH = f'../data/responses/{DATASET}/{FILE}'
VALIDATE_PATH = f'../data/validated/{DATASET}/{FILE}'

def extract_answers(ele):
    responses = ele['responses']
    answers = []
    if DATASET in ['gsm8k', 'gpqa_diamond']:
        for response in responses:
            final_answer = response.split('The final answer is')[-1].replace(':','').replace('$','').replace('\\','').replace('boxed{','').replace('}','').strip()
            answers.append(final_answer)
    elif DATASET in ['gpqa_diamond_mcq']:
        for response in responses:
            final_answer = response.split('The final answer is')[-1].replace(':','').strip()
            if len(final_answer) in [1,2]:
                option = final_answer[0]
                if option == 'a':
                    final_answer = ele['options'][0]
                elif option == 'b':
                    final_answer = ele['options'][1]
                elif option == 'c':
                    final_answer = ele['options'][2]
                elif option == 'd':
                    final_answer = ele['options'][3]
            else:
                final_answer = final_answer.replace('a)','').replace('b)','').replace('c)','').replace('d)','').strip()
            answers.append(final_answer.strip())
    else:
        raise NotImplementedError(f'Evaluation for {DATASET} is not implemented yet.')
    return answers

if __name__ == '__main__':
    with open(FILE_PATH, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    correct = 0
    variants = []
    with tqdm(total=len(data)) as pbar:
        for ele in data:
            ele['extracted_answers'] = extract_answers(ele)
            ele['variants'] = len(set(ele['extracted_answers']))
            variants.append(len(set(ele['extracted_answers'])))
            ######### GSM8k #########
            # if ele['final_answer'].strip() in ele['extracted_answers']:
            ######### GPQA-Diamond, MCQ #########
            if ele['answer'].strip() in ele['extracted_answers']:
                ele['correct'] = True
                correct += 1
            else:
                ele['correct'] = False

            with open(VALIDATE_PATH, 'a+') as f:
                f.write(json.dumps(ele)+'\n')
            pbar.update(1)
    
    print(f'For file {FILE}')
    print(f'Correct: {correct}. Total: {len(data)}. Accuracy: {correct/len(data)*100:.2f}%')
    print(f'Avg Exploration: {sum(variants)/len(variants):.2f}')