import os
import json
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    HfArgumentParser
)
from torch.utils.data import Dataset, random_split
import wandb
from huggingface_hub import login
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
import argparse

# Set up environment and login credentials
login(token='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')
os.environ['HF_HOME'] = '/data/data/dhanda/huggingface_cache'
os.environ['WANDB_API_KEY'] = '6d12fbafca76ee558ad46b08798ce77510024bbb'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@dataclass
class ScriptArguments:
    """
    Arguments for the training script
    """
    model_name: str = field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        metadata={"help": "The model name or path to be fine-tuned"}
    )
    data_path: str = field(
        default="open_math_instruct_train_10k/training_data/sft/bon_50_correct.jsonl",
        metadata={"help": "Path to the training data file (jsonl or json)"}
    )
    output_dir: str = field(
        default="./trained_models/sft/bon_fine_tuned_llama_model",
        metadata={"help": "Directory to save the fine-tuned model"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name for the W&B run"}
    )
    project_name: str = field(
        default="idea_sampling",
        metadata={"help": "W&B project name"}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for training"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for training"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate gradients before performing a backward/update pass"}
    )
    epochs: int = field(
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for the model"}
    )
    eval_split: float = field(
        default=0.2,
        metadata={"help": "Fraction of data to use for evaluation"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between evaluations"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps between logging"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between model checkpoints"}
    )
    save_limit: int = field(
        default=2,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )
    seed: int = field(
        default=21,
        metadata={"help": "Random seed for reproducibility"}
    )
    padding_side: str = field(
        default="right",
        metadata={"help": "Side to add padding ('left' or 'right')"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use mixed precision training"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bfloat16 mixed precision training (requires Ampere+ GPUs)"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training (-1 means not distributed)"}
    )


class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Only main process prints
        is_main_process = dist.get_rank() == 0 if dist.is_initialized() else True
        if is_main_process:
            print(f"Loading dataset from {file_path}...")

        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc="Loading data", disable=not is_main_process):
                    try:
                        item = json.loads(line.strip())
                        text = '<|start_header_id|>user<|end_header_id|>\n\n' + item['question'] + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' + item['response'] + '<|eot_id|>'
                        
                        encodings = self.tokenizer(
                            text, 
                            truncation=True, 
                            max_length=max_length, 
                            padding="max_length",
                            return_tensors='pt'
                        )
                        
                        self.data.append({
                            'input_ids': encodings['input_ids'].squeeze(),
                            'attention_mask': encodings['attention_mask'].squeeze(),
                            'labels': encodings['input_ids'].squeeze()  # Labels same as input for language modeling
                        })
                    except json.JSONDecodeError:
                        if is_main_process:
                            print(f"Skipping invalid JSON line: {line}")
                    except KeyError as e:
                        if is_main_process:
                            print(f"Missing key in data: {e}")
                            
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in tqdm(data, desc="Loading data", disable=not is_main_process):
                    try:
                        text = '<|start_header_id|>user<|end_header_id|>\n\n' + item['question'] + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' + item['response'] + '<|eot_id|>'
                        
                        encodings = self.tokenizer(
                            text, 
                            truncation=True, 
                            max_length=max_length, 
                            padding="max_length",
                            return_tensors='pt'
                        )
                        
                        self.data.append({
                            'input_ids': encodings['input_ids'].squeeze(),
                            'attention_mask': encodings['attention_mask'].squeeze(),
                            'labels': encodings['input_ids'].squeeze()
                        })
                    except KeyError as e:
                        if is_main_process:
                            print(f"Missing key in data: {e}")
        else:
            raise ValueError("Unsupported file format:", file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class LLMTrainer:
    def __init__(self, args):
        self.args = args
        self.is_distributed = args.local_rank != -1
        self.is_main_process = not self.is_distributed or dist.get_rank() == 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        self.training_args = None
        self.trainer = None
        
        # Set run name if not provided
        if self.args.run_name is None:
            self.args.run_name = f'bon-sft-{self.args.model_name.split("/")[-1]}'
    
    def prepare_tokenizer(self):
        if self.is_main_process:
            print(f"Loading tokenizer from {self.args.model_name}")
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, 
            add_eos_token=True, 
            add_bos_token=True,
            use_fast=True,
            padding_side=self.args.padding_side,
            trust_remote_code=True,
            use_multiprocessing=False
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.is_main_process:
                print(f"Set padding token to {self.tokenizer.pad_token}")
        
        if self.is_main_process:
            print("Loading and tokenizing dataset...")
            
        self.dataset = JSONLDataset(
            file_path=self.args.data_path, 
            tokenizer=self.tokenizer, 
            max_length=self.args.max_length
        )

        dataset_size = len(self.dataset)
        if dataset_size == 0:
            raise ValueError(f"Dataset is empty. Please check the file: {self.args.data_path}")
            
        eval_size = int(dataset_size * self.args.eval_split)
        train_size = dataset_size - eval_size
        
        # Set seed for reproducibility
        # We use a different seed for each process to avoid identical splits
        generator = torch.Generator().manual_seed(self.args.seed)
        self.train_dataset, self.eval_dataset = random_split(
            self.dataset, 
            [train_size, eval_size],
            generator=generator
        )
        
        if self.is_main_process:
            print(f"Dataset split: {train_size} training samples, {eval_size} evaluation samples")
    
    def configure_model(self):
        if self.is_main_process:
            print(f"Loading model: {self.args.model_name}")
            
        model_config = {
            'pretrained_model_name_or_path': self.args.model_name,
            'device_map': 'auto' if not self.is_distributed else None,  # In distributed mode, we'll handle device mapping differently
            'attn_implementation': 'sdpa',  # Efficient attention
            'trust_remote_code': True
        }
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(**model_config)
            if self.is_main_process:
                print("Model loaded successfully")
        except Exception as e:
            if self.is_main_process:
                print(f"Error loading model: {e}")
                print("Retrying model load with simplified config")
            # Try without some params if initial load failed
            model_config.pop('attn_implementation', None)
            self.model = AutoModelForCausalLM.from_pretrained(**model_config)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            if self.is_main_process:
                print("Gradient checkpointing enabled")
        
        # Set model to training mode
        self.model.train()
    
    def setup_training(self):
        # Initialize wandb on main process only
        if self.is_main_process:
            print(f"Initializing wandb run: {self.args.run_name}")
            wandb.init(project=self.args.project_name, name=self.args.run_name)
        else:
            # Disable wandb for other processes
            os.environ["WANDB_DISABLED"] = "true"

        # Define training arguments with distributed training settings
        self.training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epochs,
            learning_rate=self.args.learning_rate,
            logging_dir=os.path.join(self.args.output_dir, 'logs'),
            logging_steps=self.args.logging_steps,
            save_strategy='steps',
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_limit,
            eval_strategy='steps',
            eval_steps=self.args.eval_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            report_to='wandb' if self.is_main_process else 'none',
            push_to_hub=False,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            optim='adamw_torch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            remove_unused_columns=False,  # Important for custom datasets
            
            # Distributed training settings
            local_rank=self.args.local_rank,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,  # Parallel data loading
            gradient_checkpointing=True,  # Memory efficiency
            ddp_bucket_cap_mb=500  # DDP bucket size
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False  # Causal language modeling (not masked)
        )
        
        # Initialize the trainer
        if self.is_main_process:
            print("Setting up trainer")
            
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator
        )
    
    def train(self):
        if self.is_main_process:
            print("Starting training...")
            
        try:
            train_result = self.trainer.train()
            
            if self.is_main_process:
                print("Training completed!")
                
                # Save model and tokenizer (only from main process)
                print(f"Saving model to {self.args.output_dir}")
                self.trainer.save_model(self.args.output_dir)
                self.tokenizer.save_pretrained(self.args.output_dir)
                
                # Log metrics
                self.trainer.log_metrics("train", train_result.metrics)
                self.trainer.save_metrics("train", train_result.metrics)
                
                print("Running final evaluation...")
                eval_results = self.trainer.evaluate()
                self.trainer.log_metrics("eval", eval_results)
                self.trainer.save_metrics("eval", eval_results)
                
                # Finish wandb run
                wandb.finish()
                print(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")
                
        except Exception as e:
            if self.is_main_process:
                print(f"Training failed: {e}")
                import traceback
                traceback.print_exc()
                wandb.finish()
    
    def run_training_pipeline(self):
        if self.is_main_process:
            print("Starting Distributed Decoder-Only LLM Fine-Tuning Pipeline...")
            
        self.prepare_tokenizer()
        self.configure_model()
        self.setup_training()
        self.train()
        
        if self.is_main_process:
            print("Decoder-Only LLM Fine-Tuning Complete!")


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Initialize distributed training if needed
    if args.local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        
    # Set seed for all processes
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Run the trainer
    trainer = LLMTrainer(args)
    trainer.run_training_pipeline()
    
    # Clean up distributed training
    if args.local_rank != -1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()