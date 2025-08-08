# pt2bash: Portuguese to Bash Command Translator

This repository contains the complete pipeline for fine-tuning a sequence-to-sequence model that translates natural-language commands in Portuguese into functional Bash commands.

---

## 🚀 Project Overview

The goal of **pt2bash** is to bridge the gap between human language and the command line for Portuguese speakers. Leveraging a fine-tuned T5 model, pt2bash can convert a command like:

> “crie uma pasta chamada 'testes'”

into:

```bash
mkdir 'testes'
```

### Key Features

* **Preprocessing Pipeline**: Translates the English nl2bash dataset to Portuguese using a dedicated translation model.
* **Modular Training Pipeline**: Configurable scripts for fine-tuning a T5 model on the translated data.
* **Evaluation Tools**: Quantitative (BLEU score) and qualitative (interactive testing) evaluation scripts.
* **Colab Integration**: A Google Colab notebook for easy, GPU-accelerated training.

---

## 📂 Project Structure

```text
pt2bash/
├── data/              # Processed NL–Bash JSON datasets
├── models/            # Fine-tuned model artifacts (ignored by Git)
├── preprocessing/     # Dataset translation scripts
├── training/          # Modular training code
├── scripts/           # Testing and evaluation scripts
├── config.yaml        # Central project configuration
├── requirements.txt   # Python dependencies
└── README.md          # Project overview and usage
```

---

## 🛠️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/pt2bash.git
   cd pt2bash
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

The project pipeline is divided into several steps:

### 1. Build the Dataset

Translate and prepare the dataset:

```bash
python preprocessing/build_dataset.py
```

### 2. Train the Model

*In progress — add training script invocation here.*

### 3. Test & Evaluate

* **Interactive Testing**: Chat with your model.

  ```bash
  python scripts/test_model.py
  ```

* **Quantitative Evaluation**: Compute BLEU score on the test set.

  ```bash
  python scripts/evaluate_model.py
  ```

---

## 📊 Initial Results

* **Final Validation Loss**: 1.8439
* **Baseline BLEU Score**: 18.33

> This baseline demonstrates that the model is learning the translation task. Future work will focus on hyperparameter tuning and extended training to improve performance.

---
