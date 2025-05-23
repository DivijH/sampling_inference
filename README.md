# GuidedSampling: Improving Diversity for Training Large Language Models

### Directory Structure

```
├── README.md                             # This file
├── src/
│   ├── data_gen/
│   │   └── load_data.py                  # Script to download and save data from Huggingface
│   ├── keys/
│   │   └── huggingface.key               # Huggingface Key
│   ├── bon.py                            # Script for Best-of-N
│   ├── sampling.py                       # Script for Our sampling method
│   ├── eval.py                           # Script for evaluating the answers (upper bound)
├── data/
│   ├── responses/                        # Folder where all responses are stored
│   ├── validated/                        # Folder where responses as well as evalutions are stored
│   ├── gpqa_diamond.jsonl                # GPQA-Diamond dataset [Huggingface](https://huggingface.co/datasets/Idavidrein/gpqa)
│   └── gsm8k.jsonl                       # GSM8k dataset [Huggingface](https://huggingface.co/datasets/openai/gsm8k)
├── requirements.txt                      # Python dependencies
└── .gitignore
```
