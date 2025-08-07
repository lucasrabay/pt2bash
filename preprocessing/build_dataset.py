import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import os
import yaml
from pathlib import Path

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def translate_text(text, model, tokenizer, config):
    """Translates a batch of texts to Portuguese."""
    # Prepare the text for the model
    templated_text = [f">>por<< {t}" for t in text]
    
    # Tokenize the text
    inputs = tokenizer(
        templated_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=config['translation']['max_length']
    )

    # Move tensors to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate the translation
    translated_tokens = model.generate(**inputs)
    
    # Decode the tokens into text
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    
    return translated_text

def get_device(config):
    """Get the best available device based on configuration priority."""
    device_priority = config['processing']['device_priority']
    
    for device_name in device_priority:
        if device_name == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda'), 'CUDA'
        elif device_name == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps'), 'MPS'
    
    # Fallback to CPU if none of the preferred devices are available
    return torch.device('cpu'), 'CPU'

def main():
    """
    Main function to load, translate, and process the dataset.
    """
    # Load configuration
    config = load_config()
    
    # Load the translation model and tokenizer
    model_name = config['translation']['model_name']
    try:
        print(f"Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have a working internet connection and the correct model name.")
        return

    # Set up device for model
    device, device_name = get_device(config)
    print(f'Using {device_name}')

    model.to(device)
    
    # Load the dataset
    dataset_name = config['dataset']['name']
    try:
        print(f"Loading dataset: {dataset_name}")
        full_dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have a working internet connection and the correct dataset name.")
        return

    # Function to apply the translation to a batch of examples
    def translate_column(batch):
        batch['nl_pt'] = translate_text(batch['nl'], model, tokenizer, config)
        return batch

    # Create output directory
    data_dir = config['dataset']['output_dir']
    os.makedirs(data_dir, exist_ok=True)

    # iterate over each split, translate and save
    for split_name, dataset in full_dataset.items():
        print(f"\n--- Processing '{split_name}' split ---")

        print(f"Starting translation for '{split_name}' split...")
        
        translated_dataset = dataset.map(
            translate_column, 
            batched=True, 
            batch_size=config['processing']['batch_size']
        )
        
        print(f"Translation complete for '{split_name}' split.")

        # Generate output filename
        filename_template = config['output']['filename_template']
        output_filename = filename_template.format(split=split_name)
        output_path = os.path.join(data_dir, output_filename)

        # Save the translated dataset to JSON
        print(f"\nSaving translated '{split_name}' split to {output_path}...")
        translated_dataset.to_json(output_path, orient="records", lines=True)
        print(f"'{split_name}' split saved successfully.")

        # Show sample translations
        num_samples = config['output']['show_samples']
        print(f"\nSample translations for '{split_name}':")
        for i in range(min(num_samples, len(translated_dataset))):
            print(f"Original: {dataset[i]['nl']}")
            print(f"Translated: {translated_dataset[i]['nl_pt']}")
            print("-" * 50)

    print("\n--- All splits processed and saved. ---")

if __name__ == '__main__':
    main()
