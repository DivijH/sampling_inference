import json

FILE_PATH = '../../data/raw_data/olympiad_bench/test.json'
SAVE_PATH = '../../data/olympiad_bench.jsonl'

def main():
    with open(FILE_PATH, 'r') as f:
        data = json.load(f)
    
    save_data = []
    for key in ['math', 'physics']:
        for ele in data[key]:
            save_data.append(ele)
    
    with open(SAVE_PATH, 'w') as f:
        for ele in save_data:
            f.write(json.dumps(ele)+'\n')

if __name__ == '__main__':
    main()