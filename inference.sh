#!/bin/bash
echo "1. Single Message Inference Test"
python scripts/inference.py --message "I lost my credit card, please block it!"

echo ""
echo "2. Batch Evaluation on Test Set"
python scripts/inference.py --eval sample_data/test.csv --save

echo ""
echo "Inference pipeline completed successfully!"
