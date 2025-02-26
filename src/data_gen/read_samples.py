import json


DATASET = 'gsm8k'
DATASET = 'gpqa_diamond'
DATASET = 'gpqa_diamond_mcq'
INFERENCE_PATH = f'../../data/validated/{DATASET}/{DATASET}_inference.jsonl'
BON_PATH = f'../../data/validated/{DATASET}/{DATASET}_bon_10.jsonl'
SAMPLING_PATH = f'../../data/validated/{DATASET}/{DATASET}_sampling_5_2.jsonl'

INDEX = 169
BON_TRUE = False
SAMPLING_TRUE = True

def save_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_data(path):
    with open(path, 'r') as f:
        data = [json.loads(ele) for ele in f.readlines()]
    return data

def find_instance(bon, sampling, bon_true=False, sampling_true=True):
    count = 0
    ids = []
    for idx, (ele1, ele2) in enumerate(zip(bon, sampling)):
        if ele1['correct'] == bon_true and ele2['correct'] == sampling_true:
            count += 1
            ids.append(idx)
    return ids

if __name__ == '__main__':
    inference = load_data(INFERENCE_PATH)
    bon = load_data(BON_PATH)
    sampling = load_data(SAMPLING_PATH)

    ids = find_instance(bon, sampling, bon_true=BON_TRUE, sampling_true=SAMPLING_TRUE)
    print(f'For BON={BON_TRUE} ; SAMPLING={SAMPLING_TRUE} ; Total instances: {len(ids)}')
    print(f'INDEXES {ids}')

    print(f'Saving the instances at index {INDEX}')
    save_data(inference[INDEX], 'inference.json')
    save_data(bon[INDEX], 'bon.json')
    save_data(sampling[INDEX], 'sampling.json')