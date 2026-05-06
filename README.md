# Banking Intent Classification with Unsloth

This project fine-tunes a Large Language Model (Llama 3.1 8B) using the Unsloth framework to classify banking customer intents based on the `BANKING77` dataset. It uses **Strict Constrained Decoding** to guarantee that the output always matches valid intent labels without hallucinations.

## Requirements
This project included:
- **Dataset:** Uses the official `BANKING77` dataset.
- **Sampling:** Extracts a balanced subset via Stratified Sampling to accommodate computational limits (`preprocess_data.py`).
- **Data Splitting:** Preprocesses and splits data into Train, Val, and Test sets.
- **Fine-tuning:** Uses Unsloth for efficient 4-bit LoRA training.
- **Hyperparameters Documentation:** All configs (Batch size, LR, Optimizer, Epochs, Max Seq Length) are thoroughly documented in `configs/train.yaml`.
- **Inference Interface:** Strictly implements `IntentClassification(model_path)` with the `__call__` method.
- **Evaluation:** Includes a complete evaluation pipeline to report accuracy and F1 metrics on the independent Test set.

## Environment Setup

1. **Clone the repository and enter the directory**
2. **Create and activate a virtual environment (Python 3.10+):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Data Download & Training Pipeline
To automatically download the `BANKING77` dataset from HuggingFace, prepare the splits, and start the fine-tuning process, run:
```bash
bash train.sh
```
*Alternatively, run the steps manually:*
```bash
python scripts/preprocess_data.py
python scripts/train.py
```

### 2. Inference & Evaluation
To evaluate the model on the independent test set or make single predictions, run:
```bash
bash inference.sh
```
*Alternatively, run manually:*
```bash
# Evaluate the whole test set
python scripts/inference.py --eval sample_data/test.csv --save

# Predict a single message
python scripts/inference.py --message "I swallowed my card at the ATM!"
```

## Usage Example (Interface)

You can import and use the `IntentClassification` class independently in your Python code:

```python
from scripts.inference import IntentClassification

# Initialize the model using the config file path
# The config file MUST contain the path to the trained LoRA adapter (model_checkpoint)
clf = IntentClassification("configs/inference.yaml")

# Run prediction
message = "My card was charged twice for the same coffee!"
predicted_label = clf(message)

print(f"Predicted Intent: {predicted_label}")
# Expected output: transaction_charged_twice
```

## Hyperparameters & Settings
All configurations are explicitly tracked in `configs/train.yaml`. Key settings:
- **Batch Size:** 32 (per_device_train_batch_size: 32, gradient_accumulation_steps: 1)
- **Learning Rate:** 2e-4
- **Optimizer:** `adamw_8bit`
- **Training Epochs:** 4 (with Early Stopping patience: 2)
- **Maximum Sequence Length:** 256
- **Regularization:** Weight decay `0.01`, LoRA Dropout `0`

## Video link: https://drive.google.com/drive/folders/1l0_k976cXyWUJ__jiZEam9Q-U_f0vhYG?usp=drive_link
