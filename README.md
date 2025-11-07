# SnapPredict 🏈

## Overview

SnapPredict is a machine learning-based NFL game prediction system that uses historical game data to predict the winners of NFL matches. The project employs multiple machine learning models (Logistic Regression and Random Forest) to provide win probability predictions for upcoming NFL games.

## Features

- **Game Prediction**: Predicts winners of NFL games with probability estimates
- **Multiple Models**: Uses both Logistic Regression and Random Forest classifiers
- **Comprehensive Factors**: Considers various game factors including:
  - Home and away teams
  - Season type (Regular/Post season)
  - Week number
  - Stadium information
  - Weather conditions (temperature and wind)
  - Playing surface type
- **Visual Results**: Generates detailed visualization graphs for predictions
- **Model Consensus**: Provides consensus analysis when both models agree
- **Interactive Interface**: User-friendly command-line interface for predictions

## Project Structure

```
SnapPredict/
├── data/
│   └── cleaned/
│       ├── nfl_winner_predict_2019_2022.csv
│       └── processed_nfl_pass_data.csv
├── src/
│   ├── evaluate_models.py
│   ├── predict_winner.py
│   ├── prep_features.py
│   └── train_models.py
└── README.md
```

## Getting Started

1. Ensure you have Python installed with the required packages:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn

2. Clone the repository:
   ```bash
   git clone https://github.com/SidSrinivasan05/SnapPredict.git
   cd SnapPredict
   ```

3. Train the models:
   ```bash
   python src/train_models.py
   ```

4. Make predictions:
   ```bash
   python src/predict_winner.py
   ```

## How It Works

1. **Data Preparation**: Historical NFL game data is processed and cleaned
2. **Model Training**: Two models are trained on the processed data:
   - Logistic Regression: For linear probability estimation
   - Random Forest: For complex pattern recognition
3. **Prediction**: User inputs game details (teams, venue, conditions)
4. **Analysis**: Both models analyze the input and provide win probabilities
5. **Visualization**: Results are displayed with detailed probability graphs

## Model Performance

The system evaluates predictions using various metrics including accuracy and classification reports. Both models are trained on historical NFL data from 2019-2022, with the dataset split into training (80%) and testing (20%) sets.

## Output

The system provides:
- Win probability predictions from both models
- Consensus analysis when models agree
- Detailed visualization graphs
- Probability distribution charts
- Winner announcement graphics

## Authors

- Sid Srinivasan
- Manish Chakka
- Abhiram Chilakamarri
- Jyotir Sompalli

## License

[Add your license information]
