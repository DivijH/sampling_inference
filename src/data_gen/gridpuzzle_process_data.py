import json
import pandas as pd
from pprint import pprint

FILE_PATH = '../../data/raw_data/grid_puzzle/GridPuzzle.csv'
SAVE_PATH = '../../data/grid_puzzle.jsonl'

def main():
    df = pd.read_csv(FILE_PATH)
    data = df.to_dict(orient='records')

    save_data = []
    for ele in data:
        save_data.append({
            'key': ele['key'],
            'id': ele['id'],
            'question': ele['question'],
            'answer': ele['answer'],
        })
    
    with open(SAVE_PATH, 'w') as f:
        for ele in save_data:
            f.write(json.dumps(ele)+'\n')

if __name__ == '__main__':
    main()