"""
inference.py — Banking Intent Classification
  - IntentClassification  : single-message inference (required by project)
  - batch_predict          : batch inference for evaluation
  - evaluate               : full eval pipeline with metrics report

Usage (single message):
    clf = IntentClassification("configs/inference.yaml")
    print(clf("I lost my credit card"))

Usage (evaluation CLI):
    python scripts/inference.py --eval sample_data/test.csv --save
"""

import os
import json
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from unsloth import FastLanguageModel
from sklearn.metrics import classification_report, accuracy_score
from transformers import LogitsProcessor

PROMPT_TEMPLATE = """### Instruction:
Banking intent classification
 
### Input:
{}
 
### Response:
"""

def normalize_text(s: str) -> str:
    return s.strip().lower()

def extract_label(text: str) -> str:
    return text.split("### Response:")[-1].strip().split("\n")[0].strip()

def normalize_label(label: str) -> str: return label.strip().lower()

class LabelLogitsProcessor(LogitsProcessor):
    """
    Custom LogitsProcessor to strictly constrain the LLM generation 
    to a predefined set of valid labels using a Prefix-Trie approach.
    """
    def __init__(self, encoded_labels, first_tokens, eos_token_id, prompt_len):
        self.encoded_labels = encoded_labels
        self.first_tokens = first_tokens
        self.eos_token_id = eos_token_id
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            gen_toks = input_ids[i, self.prompt_len:].tolist()
            allowed = set()

            if not gen_toks:
                allowed = self.first_tokens
            else:
                for lbl in self.encoded_labels:
                    if gen_toks == lbl[:len(gen_toks)]:
                        allowed.add(lbl[len(gen_toks)] if len(gen_toks) < len(lbl) else self.eos_token_id)
                if not allowed: allowed.add(self.eos_token_id)

            mask = scores[i].new_full(scores[i].shape, float("-inf"))
            mask[list(allowed)] = 0.0
            scores[i] += mask

        return scores

class IntentClassification:
    """
    Main Interface Class for Intent Classification.
    Provides single-message inference and batch inference for evaluation.
    """
    def __init__(self, model_path: str):
        # 1. Load configuration
        with open(model_path, "r", encoding="utf-8") as f: cfg = yaml.safe_load(f)
        self.device, self.batch_size = "cuda" if torch.cuda.is_available() else "cpu", cfg.get("batch_size", 32)

        # 2. Load Model & Tokenizer
        print("Loading model and LoRA adapter...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model_checkpoint"], max_seq_length=cfg["max_seq_length"], dtype=None, load_in_4bit=cfg["load_in_4bit"]
        )
        FastLanguageModel.for_inference(self.model)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        # 3. Load valid labels
        with open(cfg["label_map_path"], "r", encoding="utf-8") as f: label_map = json.load(f)
        self.valid_labels = sorted([normalize_label(lb) for lb in label_map["label2id"]])
        print(f"Loaded {len(self.valid_labels)} labels")

        # 4. Precompute tokenized labels for constrained decoding
        self.encoded_labels = [self.tokenizer.encode(lbl, add_special_tokens=False) for lbl in self.valid_labels]
        self.first_tokens = {lbl[0] for lbl in self.encoded_labels if lbl}

    def __call__(self, message: str) -> str:
        """Single message inference."""
        inputs = self.tokenizer(PROMPT_TEMPLATE.format(message), return_tensors="pt").to(self.device)
        processor = LabelLogitsProcessor(self.encoded_labels, self.first_tokens, self.tokenizer.eos_token_id, inputs.input_ids.shape[1])
        
        outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, logits_processor=[processor], use_cache=True)
        return extract_label(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    def predict_batch(self, texts: list) -> list:
        """Batch inference for evaluation."""
        self.tokenizer.padding_side = "left"
        predictions = []

        for start in tqdm(range(0, len(texts), self.batch_size), desc="Inference"):
            prompts = [PROMPT_TEMPLATE.format(t) for t in texts[start:start + self.batch_size]]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            processor = LabelLogitsProcessor(self.encoded_labels, self.first_tokens, self.tokenizer.eos_token_id, inputs.input_ids.shape[1])

            outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=self.tokenizer.eos_token_id, logits_processor=[processor], use_cache=True)
            predictions.extend([extract_label(dec) for dec in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)])

        return predictions

def evaluate(clf, data_path, save_report=False):
    df = pd.read_csv(data_path)
    y_true, texts = df["label"].str.lower().str.strip().tolist(), df["text"].tolist()

    print(f"Evaluating: {os.path.basename(data_path)} ({len(texts)} samples)\nLabels: {len(clf.valid_labels)}\n")
    y_pred = clf.predict_batch(texts)

    correct, acc = sum(t == p for t, p in zip(y_true, y_pred)), accuracy_score(y_true, y_pred)
    
    print("=" * 60 + "\nEVALUATION REPORT\n" + "=" * 60)
    print(f"Correct : {correct} / {len(texts)}\nAccuracy: {acc * 100:.2f}%\n" + "=" * 60)
    print(classification_report(y_true, y_pred, zero_division=0))

    if save_report:
        out_path = os.path.join(os.path.dirname(data_path), "eval_predictions.csv")
        pd.DataFrame({"text": texts, "true_label": y_true, "predicted_label": y_pred, "correct": [t == p for t, p in zip(y_true, y_pred)]}).to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference.yaml")
    parser.add_argument("--message", type=str)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    clf = IntentClassification(args.config)

    if args.message:
        print("\nMessage :", args.message)
        print("Intent  :", clf(args.message))

    if args.eval:
        evaluate(clf, args.eval, save_report=args.save)