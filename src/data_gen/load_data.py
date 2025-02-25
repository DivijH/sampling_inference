import os
from pathlib import Path
import argparse
import json
import random
from tqdm import tqdm
import pandas as pd
from pprint import pprint

from datasets import load_dataset

DATASET_NAME = 'Idavidrein/gpqa'
DATASET_SUBSET = 'gpqa_diamond'
DATASET_SPLIT = 'train'
SAVE_PATH = '../../data/gpqa_diamond.jsonl'
CACHE_DIR = '/data/data/shri/Huggingface_model_cache'   # '/scratch/dhanda/huggingface_cache'
HUGGINGFACE_TOKEN = open('../keys/huggingface.key').read().strip()

random.seed(21)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, required=False, help='Dataset name from Huggingface')
    parser.add_argument('--dataset_subset', type=str, default=DATASET_SUBSET, required=False, help='Subset of the dataset')
    parser.add_argument('--dataset_split', type=str, default=DATASET_SPLIT, required=False, help='Split of the subset')
    parser.add_argument('--huggingface_token', type=str, default=HUGGINGFACE_TOKEN, required=False, help='Hugging Face API token')
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR, required=False, help='Cache directory')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH, required=False, help='File where the dataset will be stored')
    return parser.parse_args()

class LoadDataset:
    def __init__(
        self,
        dataset_name,
        dataset_subset,
        dataset_split,
        save_path,
        huggingface_token = None,
        cache_dir = None,
    ):
        if os.path.exists(save_path):
            input(f'{save_path} already exists. Press Enter to overwrite or Ctrl+C to exit.')
            os.remove(save_path)

        self.dataset = load_dataset(
            dataset_name,
            dataset_subset,
            split = dataset_split,
            token = huggingface_token,
            cache_dir = cache_dir
        )
        self.save_path = save_path
    
    def _process_instance(self, ele):
        data_ele = {
            'question': ele['Question'],
            'options': random.sample([
                ele['Correct Answer'],
                ele['Incorrect Answer 1'],
                ele['Incorrect Answer 2'],
                ele['Incorrect Answer 3'],
            ], 4),
            'answer': ele['Correct Answer'],
        }
        return data_ele
    
    def _save_data(self):
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        extension = self.save_path.split('.')[-1]
        if extension == 'jsonl':
            for ele in self.data:
                with open(self.save_path, 'a+') as f:
                    f.write(json.dumps(ele)+'\n')
        else:
            raise NotImplementedError(f'{extension} not supported yet.')

    def process_and_save(self):
        self.data = []
        with tqdm(total=len(self.dataset)) as pbar:
            for ele in self.dataset:
                self.data.append(self._process_instance(ele))
                pbar.update(1)
        print(f'Processing finished. Saving data in {self.save_path}...')
        self._save_data()


def main():
    args = get_args()

    local_data = LoadDataset(
        dataset_name = args.dataset_name,
        dataset_subset = args.dataset_subset,
        dataset_split = args.dataset_split,
        save_path = args.save_path,
        huggingface_token = args.huggingface_token,
        cache_dir = args.cache_dir
    )
    local_data.process_and_save()

if __name__ == '__main__':
    main()