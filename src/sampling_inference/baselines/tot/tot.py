import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from vllm_inference import VLLMInferenceBase
from pathlib import Path
from tqdm import tqdm
from abc import abstractmethod
import re


class ToTInferenceBase(VLLMInferenceBase):
    @abstractmethod
    def thought_generation_prompt(self, ele, previous_thoughts=""):
        pass
    
    @abstractmethod
    def thought_evaluation_prompt(self, ele, thought_path):
        pass
    
    @abstractmethod
    def final_solution_prompt(self, ele, thought_path):
        pass
    
    def extract_score(self, response):
        text = response.strip()
        # 1) Prefer a line that contains only the score
        m = re.search(r'(?im)^\s*(?:score|rating)?\s*[:\-]?\s*(10|[1-9])\s*$', text)
        if m:
            return int(m.group(1))
        # 2) Otherwise take the LAST 1–10 we see
        numbers = re.findall(r'(?:^|\D)(10|[1-9])(?!\d)', text)
        if numbers:
            return int(numbers[-1])
        return 5  # neutral fallback

    def tot(self, input_file, output_file, number_thoughts=5, number_steps=3, top_k=2, number_final_solutions=10, index=0):
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            data = self._load_data(input_file, index)
            
            all_results = []
            
            for data_ele in tqdm(data, desc="Processing with Tree of Thought"):
                # Track all reasoning paths as a tree
                # Each path is a tuple: (thought_sequence, cumulative_score)
                current_paths = [("", 0.0)]  # Start with empty reasoning path
                
                # Generate thoughts layer by layer (BFS approach)
                for step in range(number_steps):
                    next_paths = []
                    
                    for path_text, path_score in current_paths:
                        # Generate k thoughts for this path
                        prompts = [self.thought_generation_prompt(data_ele, path_text) for _ in range(number_thoughts)]
                        thoughts = self.get_responses(prompts, no_responses=1)
                        thoughts = [t[0] for t in thoughts]  # Flatten
                        
                        # Evaluate each thought
                        for thought in thoughts:
                            # Construct new path
                            if path_text:
                                new_path = f"{path_text}\n\nStep {step + 1}:\n{thought}"
                            else:
                                new_path = f"Step 1:\n{thought}"
                            
                            # Evaluate the new path
                            eval_prompt = self.thought_evaluation_prompt(data_ele, new_path)
                            eval_response = self.get_responses([eval_prompt], no_responses=1)[0][0]
                            score = self.extract_score(eval_response)
                            
                            # Calculate cumulative score (average)
                            new_cumulative_score = (path_score * step + score) / (step + 1)
                            next_paths.append((new_path, new_cumulative_score))
                    
                    # Select top-k paths to continue exploring
                    next_paths.sort(key=lambda x: x[1], reverse=True)
                    current_paths = next_paths[:top_k]
                
                # Generate final solutions from the best reasoning paths
                best_paths = current_paths[:min(len(current_paths), top_k)]
                final_solution_prompts = []
                
                for path_text, path_score in best_paths:
                    # Generate multiple final solutions from each good path
                    num_solutions_per_path = max(1, number_final_solutions // len(best_paths))
                    for _ in range(num_solutions_per_path):
                        final_solution_prompts.append(self.final_solution_prompt(data_ele, path_text))
                
                # Get final solutions
                final_solutions = self.get_responses(final_solution_prompts, no_responses=1)
                final_solutions = [s[0] for s in final_solutions]
                
                # Store results
                data_ele['reasoning_paths'] = [{"path": path, "score": score} for path, score in best_paths]
                data_ele['responses'] = final_solutions
                data_ele['tot_params'] = {
                    'number_thoughts': number_thoughts,
                    'number_steps': number_steps,
                    'top_k': top_k,
                    'number_final_solutions': number_final_solutions
                }
                all_results.append(data_ele)
            
            self._save_data(output_file, all_results)
        finally:
            self.cleanup()
    
    def rs_prompt(self, ele):
        return None
    
    def gs_initial_prompt(self, ele):
        return None
    
    def gs_more_prompt(self, ele, ideas_text):
        return None
    
    def gs_prompt(self, ele, idea):
        return None
