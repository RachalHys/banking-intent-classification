import os
import yaml
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer                      
from transformers import EarlyStoppingCallback
from unsloth.chat_templates import train_on_responses_only
import warnings

warnings.filterwarnings("ignore")

PROMPT_TEMPLATE = """### Instruction:
Banking intent classification

### Input:
{}

### Response:
"""

def main():
    """
    Main function to fine-tune the generative Llama model for Intent Classification.
    Uses Unsloth for fast 4-bit LoRA training and TRL's SFTTrainer.
    """
    print("Initialize Model & Tokenizer")
    with open("configs/train.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    EOS_TOKEN = tokenizer.eos_token

    def format_prompt(examples):
        """Format the dataset rows into the specific Prompt Template format."""
        return {"text": [PROMPT_TEMPLATE.format(msg) + lbl + EOS_TOKEN for msg, lbl in zip(examples["text"], examples["label"])]}

    print("Load and Format dataset")
    df_train, df_val = pd.read_csv(f"{config['data_dir']}/train.csv"), pd.read_csv(f"{config['data_dir']}/val.csv")
    train_dataset, val_dataset = Dataset.from_pandas(df_train).map(format_prompt, batched=True), Dataset.from_pandas(df_val).map(format_prompt, batched=True)

    print("Training...")
    args = SFTConfig(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim=config["optim"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        seed=config["seed"],
        output_dir=config["output_dir"],
        report_to="none",
        eval_strategy=config.get("eval_strategy", "steps"),
        save_strategy=config.get("save_strategy", "steps"),
        logging_strategy=config.get("logging_strategy", "steps"),
        eval_steps=config.get("eval_steps", 100),
        save_steps=config.get("save_steps", 100), 
        logging_steps=config.get("logging_steps", 100),
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        save_total_limit=config.get("save_total_limit", 3),
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=False,
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.get("early_stopping_patience", 2))],
    )

    # Using Unsloth's recommended utility to mask prompt tokens from loss calculation
    trainer = train_on_responses_only(trainer, instruction_part="### Instruction:\n", response_part="### Response:\n")

    trainer.train()

    print("Save Best Model")
    final_dir = config["final_model_dir"]
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"LoRA adapters saved at: {final_dir}")

if __name__ == "__main__":
    main()
