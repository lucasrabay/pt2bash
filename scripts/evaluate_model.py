import os
import sys
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
eval_cfg = cfg['evaluation']

# Configuration values
MODEL_PATH = eval_cfg['model_path']
DATA_PATH = eval_cfg['data_path']
TASK_PREFIX = eval_cfg['task_prefix']
BATCH_SIZE = eval_cfg['batch_size']
MAX_NEW_TOKENS = eval_cfg.get('max_new_tokens', 128)

def select_device(priorities):
    for dev in priorities:
        if dev == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if dev == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        if dev == 'cpu':
            return 'cpu'
    return 'cpu'

device = select_device(cfg['processing']['device_priority'])
print(f"Using device: {device}")

# LOAD EVERYTHING
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)  # type: ignore
test_dataset = load_dataset('json', data_files=DATA_PATH)['train']

# Load the BLEU metric calculator
bleu_metric = evaluate.load('bleu')

predictions = []
references = []

print(f"\nGenerating predictions for {len(test_dataset)} test examples...")

# Batch inference
total = len(test_dataset)
for i in tqdm(range(0, total, BATCH_SIZE), desc="Evaluating"):  # type: ignore
    batch = test_dataset[i : i + BATCH_SIZE]
    prompts = [TASK_PREFIX + text for text in batch['nl_pt']]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions.extend(decoded_preds)
    references.extend([[ref] for ref in batch['bash']])

# CALCULATE BLEU SCORE
print("\nCalculating BLEU score...")
results = bleu_metric.compute(predictions=predictions, references=references)

print("\n--- EVALUATION RESULTS ---")
print(f"BLEU Score: {results['bleu'] * 100:.2f}")
print("--------------------------")
print("\nReminder: BLEU is a score from 0 to 100. Higher is better.")
print("A good baseline for this kind of task would be in the 30-50 range.")
