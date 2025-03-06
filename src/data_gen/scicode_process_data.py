import json


FILE_PATH = '../../data/raw_data/scicode/problems_all.jsonl'

def main():
    with open(FILE_PATH, 'r') as f:
        data = [json.loads(ele) for ele in f.readlines()]
    
    for ele in data:
        if ele['problem_id'] == '13':
            print('Found')
            with open('tmp.json', 'w') as f:
                json.dump(ele, f, indent=4)
    
    # print(data[0].keys())
    # print('*'*50)
    # print(data[0]['problem_description_main'])
    # print('*'*50)
    # print(data[0]['problem_io'])
    # print('*'*50)
    # print(data[0]['required_dependencies'])
    # print('*'*50)
    

if __name__ == '__main__':
    main()