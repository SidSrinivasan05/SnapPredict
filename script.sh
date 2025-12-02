#!/bin/bash
echo "Step 1: Preparing features..."
python src/prep_features.py

echo "Step 2: Training models..."
python src/train_models.py

echo "Step 3: Evaluating models..."
python src/evaluate_models.py

echo "Pipeline complete! Run predict_winner.py for predictions."