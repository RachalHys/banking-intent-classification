#!/bin/bash
echo "====================================="
echo "1. Single Message Inference Test"
echo "Executing command: python scripts/inference.py --message \"I lost my credit card, please block it!\""
echo "====================================="
python scripts/inference.py --message "I lost my credit card, please block it!"

echo ""
echo "====================================="
echo "2. Batch Evaluation on Test Set"
echo "Executing command: python scripts/inference.py --eval sample_data/test.csv --save"
echo "====================================="
python scripts/inference.py --eval sample_data/test.csv --save

echo ""
echo "Inference pipeline completed successfully!"
