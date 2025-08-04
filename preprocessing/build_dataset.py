import torch
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import os

def translate_text(text, model, tokenizer):
    """Translates a batch of texts to Portuguese."""
    # Prepare the text for the model
    templated_text = [f">>por<< {t}" for t in text]
    
    # Tokenize the text
    inputs = tokenizer(templated_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move tensors to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate the translation
    translated_tokens = model.generate(**inputs)
    
    # Decode the tokens into text
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    
    return translated_text

def main():
    """
    Main function to load, translate, and process the dataset.
    """
    # Load the translation model and tokenizer
    model_name = 'Helsinki-NLP/opus-mt-tc-big-en-pt'
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have a working internet connection and the correct model name.")
        return

    # Set up device for model
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model.to(device)
    
    # Load the dataset
    try:
        full_dataset = load_dataset("jiacheng-ye/nl2bash")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have a working internet connection and the correct dataset name.")
        return

    # Function to apply the translation to a batch of examples
    def translate_column(batch):
        batch['nl_pt'] = translate_text(batch['nl'], model, tokenizer)
        return batch


    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # iterate over each split, translate and save
    for split_name, dataset in full_dataset.items():
        print(f"\n--- Processing '{split_name}' split ---")

        print(f"Starting translation for '{split_name}' split...")
        translated_dataset = dataset.map(translate_column, batched=True, batch_size=16)
        print(f"Translation complete for '{split_name}' split.")

        output_filename = f"pt2bash_{split_name}.json"
        output_path = os.path.join(data_dir, output_filename)

        # Save the translated dataset to a JSON file
        print(f"\nSaving translated '{split_name}' split to {output_path}...")
        translated_dataset.to_json(output_path, orient="records", lines=True)
        print(f"'{split_name}' split saved successfully.")

    print("\n--- All splits processed and saved. ---")

if __name__ == '__main__':
    main()
