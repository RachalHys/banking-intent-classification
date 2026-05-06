#!/bin/bash
echo "1. Data Preparation and Sampling"
python scripts/preprocess_data.py

echo ""
echo "2. Fine-tuning LLama 3.1 8B with Unsloth"
python scripts/train.py

echo ""