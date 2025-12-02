# SnapPredict 🏈

## Overview

SnapPredict is a machine learning-based NFL game prediction system that uses historical game data to predict the winners of NFL matches. The project employs multiple machine learning models (Logistic Regression and Random Forest) to provide win probability predictions for upcoming NFL games.

## Features

- **Game Prediction**: Predicts winners of NFL games with probability estimates
- **Multiple Models**: Uses both Logistic Regression and Random Forest classifiers
- **Comprehensive Parameters**:
  - Home and away teams
  - Season type (Regular/Post season)
  - Week number
  - Stadium information
  - Weather conditions (temperature and wind)
  - Playing surface type
- **Visual Results**: Generates detailed visualization graphs for predictions
- **Model Consensus**: Provides a consensus analysis when both models agree
- **Interactive Interface**: Made a User-friendly command-line interface for predictions

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
├── output/
│   ├── models/
│   │   ├── features.csv
│   │   ├── logistic_model.pkl
│   │   ├── rf_model.pkl
│   │   ├── encoders.pkl
│   │   └── test_data.pkl
│   └── predictions/
│       ├── prediction_*.png
│       └── winner_*.png
└── README.md
```

## Directory Structure Details

### Data Directory
/data/: Contains all raw and cleaned datasets

/data/cleaned/: Holds our processed CSV files ready for model use
- /data/cleaned/nfl_winner_predict_2019_2022.csv: Historical NFL game data used for training
- /data/cleaned/processed_nfl_pass_data.csv: Additional NFL passing statistics

### Source Code
/src/: Main code directory for the project
- /src/prep_features.py: Prepares and cleans data for model training
- /src/train_models.py: Creates and saves our prediction models
- /src/evaluate_models.py: Tests how well our models perform
- /src/predict_winner.py: Makes predictions for new NFL games

### Output Directory
/output/: Where we store all generated files

/output/models/: Trained models and related files
- /output/models/features.csv: Final dataset used for training
- /output/models/logistic_model.pkl: Our logistic regression model
- /output/models/rf_model.pkl: Our random forest model
- /output/models/encoders.pkl: Tools for converting team names and other text data
- /output/models/test_data.pkl: Data we use to test model accuracy

/output/predictions/: Visual results from our predictions
- /output/predictions/prediction_*.png: Charts showing win probabilities
- /output/predictions/winner_*.png: Graphics announcing predicted winners

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/SidSrinivasan05/SnapPredict.git
   cd SnapPredict
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
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

The system evaluates predictions using various metrics including accuracy and classification reports. Both models are trained on historical NFL data from 2019-2022, with the dataset split into training (80%) and testing (20%) sets (80/20).

## Output

The system we made provides:
- Win probability predictions from both models
- Consensus analysis when models agree
- Visualization graphs
- Probability distribution charts
- Winner announced!

## Authors

- Sid Srinivasan
- Manish Chakka
- Abhiram Chilakamarri
- Jyotir Sompalli
