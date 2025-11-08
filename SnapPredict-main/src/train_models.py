# src/train_model.py
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

FEATURES_CSV = "output/models/features.csv"
OUT_DIR = "output/models"
LOGISTIC_MODEL = os.path.join(OUT_DIR, "logistic_model.pkl")
RF_MODEL = os.path.join(OUT_DIR, "rf_model.pkl")
ENCODERS_FILE = os.path.join(OUT_DIR, "encoders.pkl")
TEST_DATA_FILE = os.path.join(OUT_DIR, "test_data.pkl")

def encode_features(df):
    """Encode categorical features and return encoders"""
    encoders = {}
    categorical_cols = ["home_team", "away_team", "season_type", "stadium", "roof", "surface"]
    
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].fillna('Unknown')
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    
    return df_encoded, encoders

def prepare_data(df):
    """Prepare features and target for modeling"""
    X = df.drop(columns=["game_id", "home_win"], errors="ignore")
    y = df["home_win"]
    
    numeric_cols = ["week", "temp", "wind"]
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    return X, y

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression model"""
    print("\n=== Training Logistic Regression ===")
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
    lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test))
    
    print(f"Train Accuracy: {lr_train_acc:.4f}")
    print(f"Test Accuracy: {lr_test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, lr_model.predict(X_test), 
                                target_names=["Away Win", "Home Win"]))
    
    with open(LOGISTIC_MODEL, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"Model saved: {LOGISTIC_MODEL}")
    
    return lr_model

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model"""
    print("\n=== Training Random Forest ===")
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))
    
    print(f"Train Accuracy: {rf_train_acc:.4f}")
    print(f"Test Accuracy: {rf_test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, rf_model.predict(X_test),
                                target_names=["Away Win", "Home Win"]))
    
    with open(RF_MODEL, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Model saved: {RF_MODEL}")
    
    return rf_model

def load_and_prepare_data():
    """Load CSV and prepare train/test splits"""
    print("Loading features...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} games")
    
    print("Encoding categorical features...")
    df_encoded, encoders = encode_features(df)
    
    with open(ENCODERS_FILE, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved: {ENCODERS_FILE}")
    
    X, y = prepare_data(df_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    with open(TEST_DATA_FILE, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": X.columns.tolist()
        }, f)
    print(f"Test data saved: {TEST_DATA_FILE}")
    
    return X_train, X_test, y_train, y_test

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    lr_model = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60)

if __name__ == "__main__":
    main()