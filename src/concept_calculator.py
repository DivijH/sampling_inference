import os
import json

MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
DATASET = 'olympiad_bench'
INPUT_FILE = f'../data/concepts/{MODEL_NAME.split("/")[-1]}/{DATASET}_rs_100_concepts.jsonl'


def main():
    with open(INPUT_FILE, 'r') as f:
        data = [json.loads(ele) for ele in f.readlines()]
    
    
    concept_count = []
    for ele in data:
        concepts = 0
        seen_concepts = set()
        for concept in ele['concepts']:
            for seen_concept in seen_concepts:
                if seen_concept not in concept and concept not in seen_concept:
                    concepts += 1
            seen_concepts.add(concept)
        concept_count.append(concepts)
            
    
    print(f'Average number of concepts: {sum(concept_count) / len(data)}')

if __name__ == '__main__':
    main()