from pathlib import Path
import json
from tqdm import tqdm

SAVE = False
# MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
# MODEL_NAME = 'iaa_fine_tuned_llama_model'
DATASET = 'csqa'

# FILE = f'{DATASET}_rs_100.json'
FILE = f'{DATASET}_gs_5_20.json'
# FILE = f'{DATASET}_tot_5_3_4.json'

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
    if isinstance(ele['responses'][0], list):    # For GS
        responses = [resp for idea_responses in ele['responses'] for resp in idea_responses]
    else: # For RS
        responses = ele['responses']
    answers = []
    for response in responses:
        try:
            final_answer = response.split('**Final Answer**')[1].strip()
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
                final_answer = final_answer.strip()
        except Exception as e:
            final_answer = ''
        answers.append(final_answer.strip())
    return answers

def is_instance_correct(ele):
    ele['correctness'] = []
    for answer in ele['extracted_answers']:
        ele['correctness'].append(ele['answer'].lower().strip() in answer)
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