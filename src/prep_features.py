# src/prep_features.py
import os
import pandas as pd

RAW_PATH = "data/cleaned/nfl_winner_predict_2019_2022.csv"
OUT_DIR = "output/models"
OUT_CSV = os.path.join(OUT_DIR, "features.csv")

USE_COLS = [
    "game_id",
    "home_team", "away_team",
    "season_type", "week",
    "stadium", "roof", "surface",
    "temp", "wind",
    "total_home_score", "total_away_score",
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_PATH)

    missing = [c for c in USE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df[USE_COLS].copy()

    df = df.sort_values("game_id").drop_duplicates(subset=["game_id"], keep="first")

    df["home_win"] = (df["total_home_score"] > df["total_away_score"]).astype(int)

    for col in ["week", "temp", "wind"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    model_cols = [
        "game_id",                
        "home_team", "away_team",
        "season_type", "week",
        "stadium", "roof", "surface",
        "temp", "wind",
        "home_win",                # target
    ]
    df = df[model_cols]

    df = df.dropna(subset=["home_team", "away_team", "week", "home_win"])

    df.to_csv(OUT_CSV, index=False)
    print(f"Created feature file: {OUT_CSV}")

if __name__ == "__main__":
    main()