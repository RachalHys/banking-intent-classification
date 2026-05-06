import os
import json
import yaml
import random
import numpy as np
import pandas as pd
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "sampling.yaml")


def clean_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def sampling_data(df, total_per_class, train_ratio, seed):
    """
    Perform stratified sampling and split the dataset into train and validation sets.
    Ensures balanced distribution up to 'total_per_class' samples per class.
    """
    train_parts = []
    val_parts = []

    for label, group in df.groupby("label"):
        group = group.sample(frac=1, random_state=seed + int(label))

        n_total = min(len(group), total_per_class)
        n_train = min(round(n_total * train_ratio), n_total)
        group = group.iloc[:n_total]

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df


def main():
    """Main pipeline for loading, cleaning, sampling, and saving the dataset."""
    # 1. Load configuration
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    total_n, train_ratio = cfg["sampling"]["total_per_class"], cfg["sampling"]["train_ratio"]
    out_dir = os.path.join(BASE_DIR, cfg["paths"]["output_dir"])

    random.seed(seed)
    np.random.seed(seed)

    # 2. Load the official HuggingFace dataset
    dataset = load_dataset(cfg["dataset_name"])
    df_train, df_test = dataset["train"].to_pandas(), dataset["test"].to_pandas()

    # 3. Clean and normalize text
    df_train["text"], df_test["text"] = clean_text_series(df_train["text"]), clean_text_series(df_test["text"])

    df_train, df_test = df_train[df_train["text"] != ""], df_test[df_test["text"] != ""]

    # 4. Extract label taxonomy
    names = dataset["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in id2label.items()}

    # 5. Perform stratified sampling and splitting
    train_df, val_df = sampling_data(df_train, total_per_class=total_n, train_ratio=train_ratio, seed=seed)

    # 6. Shuffle the test set
    test_df = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 7. Map numerical labels back to descriptive string labels
    for df in (train_df, val_df, test_df):
        df["label"] = df["label"].map(id2label)

    # 8. Save the processed data and label mappings
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=4)

    train_df[["text", "label"]].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df[["text", "label"]].to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df[["text", "label"]].to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()