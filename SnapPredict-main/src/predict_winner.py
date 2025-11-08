# src/predict_winner.py
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

OUT_DIR = "output/models"
GRAPHS_DIR = "output/predictions"
LOGISTIC_MODEL = os.path.join(OUT_DIR, "logistic_model.pkl")
RF_MODEL = os.path.join(OUT_DIR, "rf_model.pkl")
ENCODERS_FILE = os.path.join(OUT_DIR, "encoders.pkl")
FEATURES_CSV = "output/models/features.csv"

def load_models_and_encoders():
    """Load trained models and encoders"""
    with open(LOGISTIC_MODEL, "rb") as f:
        lr_model = pickle.load(f)
    
    with open(RF_MODEL, "rb") as f:
        rf_model = pickle.load(f)
    
    with open(ENCODERS_FILE, "rb") as f:
        encoders = pickle.load(f)
    
    return lr_model, rf_model, encoders

def get_valid_options(encoder, field_name):
    """Get valid options for a categorical field"""
    if field_name in encoder:
        options = list(encoder[field_name].classes_)
        options = [opt for opt in options if opt != 'Unknown']
        return sorted(options)
    return []

def get_user_input(encoders):
    """Prompt user for game details"""
    print("\n" + "="*60)
    print("NFL GAME WINNER PREDICTION")
    print("="*60)
    
    teams = get_valid_options(encoders, "home_team")
    print(f"\nAvailable teams: {', '.join(teams[:10])}... ({len(teams)} total)")
    
    home_team = input("\nEnter HOME team: ").strip()
    away_team = input("Enter AWAY team: ").strip()
    
    season_types = get_valid_options(encoders, "season_type")
    print(f"\nSeason types: {', '.join(season_types)}")
    season_type = input("Enter season type (e.g., REG, POST): ").strip() or "REG"
    
    week = input("Enter week number (1-18): ").strip()
    week = int(week) if week else 1
    
    stadiums = get_valid_options(encoders, "stadium")
    print(f"\nAvailable stadiums: {', '.join(stadiums[:5])}... ({len(stadiums)} total)")
    stadium = input("Enter stadium name (press Enter for home team's stadium): ").strip()
    if not stadium:
        stadium = "Unknown"
    
    roofs = get_valid_options(encoders, "roof")
    print(f"\nRoof types: {', '.join(roofs)}")
    roof = input("Enter roof type (e.g., outdoors, dome, retractable): ").strip() or "outdoors"
    
    surfaces = get_valid_options(encoders, "surface")
    print(f"\nSurface types: {', '.join(surfaces)}")
    surface = input("Enter surface type (e.g., grass, fieldturf): ").strip() or "grass"
    
    temp = input("\nEnter temperature (°F, press Enter to skip): ").strip()
    temp = float(temp) if temp else None
    
    wind = input("Enter wind speed (mph, press Enter to skip): ").strip()
    wind = float(wind) if wind else None
    
    game_data = {
        "home_team": home_team,
        "away_team": away_team,
        "season_type": season_type,
        "week": week,
        "stadium": stadium,
        "roof": roof,
        "surface": surface,
        "temp": temp,
        "wind": wind
    }
    
    return game_data

def encode_game_data(game_data, encoders):
    """Encode user input using trained encoders"""
    encoded_data = {}
    
    df = pd.read_csv(FEATURES_CSV)
    
    categorical_cols = ["home_team", "away_team", "season_type", "stadium", "roof", "surface"]
    
    for col in categorical_cols:
        value = game_data.get(col, "Unknown")
        if col in encoders:
            try:
                encoded_data[col] = encoders[col].transform([value])[0]
            except ValueError:
                print(f"Warning: '{value}' not found in {col}, using default")
                encoded_data[col] = 0
    
    encoded_data["week"] = game_data.get("week", 1)
    encoded_data["temp"] = game_data.get("temp") if game_data.get("temp") else df["temp"].median()
    encoded_data["wind"] = game_data.get("wind") if game_data.get("wind") else df["wind"].median()
    
    return encoded_data

def predict_winner(lr_model, rf_model, encoded_data, game_data):
    """Make predictions using both models"""
    feature_order = ["home_team", "away_team", "season_type", "week", "stadium", "roof", "surface", "temp", "wind"]
    X = np.array([[encoded_data[f] for f in feature_order]])
    
    lr_pred = lr_model.predict(X)[0]
    lr_proba = lr_model.predict_proba(X)[0]
    
    rf_pred = rf_model.predict(X)[0]
    rf_proba = rf_model.predict_proba(X)[0]
    
    results = {
        "lr_winner": game_data["home_team"] if lr_pred == 1 else game_data["away_team"],
        "lr_home_prob": lr_proba[1] * 100,
        "lr_away_prob": lr_proba[0] * 100,
        "rf_winner": game_data["home_team"] if rf_pred == 1 else game_data["away_team"],
        "rf_home_prob": rf_proba[1] * 100,
        "rf_away_prob": rf_proba[0] * 100,
    }
    
    return results

def display_results(game_data, results):
    """Display prediction results in terminal"""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nMatchup: {game_data['away_team']} @ {game_data['home_team']}")
    print(f"Week {game_data['week']} - {game_data['season_type']}")
    print(f"Stadium: {game_data['stadium']} ({game_data['roof']}, {game_data['surface']})")
    if game_data.get('temp'):
        print(f"Weather: {game_data['temp']}°F, {game_data.get('wind', 0)} mph wind")
    
    print("\n" + "-"*60)
    print("LOGISTIC REGRESSION MODEL")
    print("-"*60)
    print(f"Predicted Winner: {results['lr_winner']}")
    print(f"  {game_data['home_team']} (Home): {results['lr_home_prob']:.1f}% win probability")
    print(f"  {game_data['away_team']} (Away): {results['lr_away_prob']:.1f}% win probability")
    
    print("\n" + "-"*60)
    print("RANDOM FOREST MODEL")
    print("-"*60)
    print(f"Predicted Winner: {results['rf_winner']}")
    print(f"  {game_data['home_team']} (Home): {results['rf_home_prob']:.1f}% win probability")
    print(f"  {game_data['away_team']} (Away): {results['rf_away_prob']:.1f}% win probability")
    
    print("\n" + "="*60)
    if results['lr_winner'] == results['rf_winner']:
        print(f"CONSENSUS PICK: {results['lr_winner']}")
        avg_prob = (results['lr_home_prob'] if results['lr_winner'] == game_data['home_team'] 
                   else results['lr_away_prob'] + results['rf_home_prob'] if results['rf_winner'] == game_data['home_team'] 
                   else results['rf_away_prob']) / 2
        print(f"   Average Win Probability: {avg_prob:.1f}%")
    else:
        print("MODELS DISAGREE")
        print(f"   Logistic Regression picks: {results['lr_winner']}")
        print(f"   Random Forest picks: {results['rf_winner']}")
    print("="*60)

def plot_prediction_graphs(game_data, results):
    """Generate visualization graphs for the prediction"""
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    
    teams = [game_data['home_team'], game_data['away_team']]
    lr_probs = [results['lr_home_prob'], results['lr_away_prob']]
    rf_probs = [results['rf_home_prob'], results['rf_away_prob']]
    
    x = np.arange(len(teams))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, lr_probs, width, label='Logistic Regression', 
                    color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rf_probs, width, label='Random Forest',
                    color='forestgreen', alpha=0.8)
    
    ax1.set_ylabel('Win Probability (%)', fontsize=12)
    ax1.set_title(f'Win Probability Predictions\n{game_data["away_team"]} @ {game_data["home_team"]}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t}\n(Home)" if t == game_data['home_team'] else f"{t}\n(Away)" 
                         for t in teams])
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    ax2 = fig.add_subplot(gs[1, 0])
    
    colors = ['#4CAF50' if results['lr_winner'] == game_data['home_team'] else '#FF9800',
              '#FF9800' if results['lr_winner'] == game_data['home_team'] else '#4CAF50']
    
    wedges, texts, autotexts = ax2.pie(
        [results['lr_home_prob'], results['lr_away_prob']],
        labels=[f"{game_data['home_team']}\n(Home)", f"{game_data['away_team']}\n(Away)"],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )
    ax2.set_title('Logistic Regression\nPrediction', fontsize=12, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, 1])
    
    colors = ['#4CAF50' if results['rf_winner'] == game_data['home_team'] else '#FF9800',
              '#FF9800' if results['rf_winner'] == game_data['home_team'] else '#4CAF50']
    
    wedges, texts, autotexts = ax3.pie(
        [results['rf_home_prob'], results['rf_away_prob']],
        labels=[f"{game_data['home_team']}\n(Home)", f"{game_data['away_team']}\n(Away)"],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )
    ax3.set_title('Random Forest\nPrediction', fontsize=12, fontweight='bold')
    
    game_info = f"Week {game_data['week']} - {game_data['season_type']}\n"
    game_info += f"{game_data['stadium']} ({game_data['roof']}, {game_data['surface']})"
    if game_data.get('temp'):
        game_info += f"\n{game_data['temp']}°F, {game_data.get('wind', 0)} mph wind"
    
    fig.text(0.5, 0.02, game_info, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    filename = f"prediction_{game_data['away_team']}_at_{game_data['home_team']}_{timestamp}.png"
    output_path = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Prediction graphs saved: {output_path}")
    plt.close()
    

    create_winner_graphic(game_data, results, timestamp)

def create_winner_graphic(game_data, results, timestamp):
    """Create a bold winner announcement graphic"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    consensus = results['lr_winner'] == results['rf_winner']
    winner = results['lr_winner'] if consensus else "SPLIT DECISION"
    
    fig.patch.set_facecolor('#1a1a2e' if consensus else '#333333')
    
    title_text = "PREDICTED WINNER" if consensus else "SPLIT DECISION"
    ax.text(0.5, 0.85, title_text, fontsize=32, fontweight='bold',
            ha='center', va='center', color='white')

    if consensus:
        ax.text(0.5, 0.65, winner, fontsize=48, fontweight='bold',
                ha='center', va='center', color='#4CAF50')
        

        avg_prob = (results['lr_home_prob'] if winner == game_data['home_team'] 
                   else results['lr_away_prob'] + 
                   results['rf_home_prob'] if winner == game_data['home_team'] 
                   else results['rf_away_prob']) / 2
        
        ax.text(0.5, 0.50, f"{avg_prob:.1f}% Win Probability", fontsize=24,
                ha='center', va='center', color='white')
    else:
        ax.text(0.5, 0.65, f"LR: {results['lr_winner']}", fontsize=28,
                ha='center', va='center', color='#2196F3')
        ax.text(0.5, 0.50, f"RF: {results['rf_winner']}", fontsize=28,
                ha='center', va='center', color='#4CAF50')
    

    matchup = f"{game_data['away_team']} @ {game_data['home_team']}"
    ax.text(0.5, 0.30, matchup, fontsize=20, ha='center', va='center', color='white')
    
    details = f"Week {game_data['week']} • {game_data['season_type']} • {game_data['stadium']}"
    ax.text(0.5, 0.20, details, fontsize=14, ha='center', va='center', color='lightgray')
    

    agreement_text = "Both models agree" if consensus else "Models disagree - use caution"
    ax.text(0.5, 0.10, agreement_text, fontsize=12, ha='center', va='center',
            color='#4CAF50' if consensus else '#FF9800', style='italic')
    
    filename = f"winner_{game_data['away_team']}_at_{game_data['home_team']}_{timestamp}.png"
    output_path = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Winner graphic saved: {output_path}")
    plt.close()

def main():
    print("Loading models...")
    lr_model, rf_model, encoders = load_models_and_encoders()
    
    game_data = get_user_input(encoders)
    
    print("\nProcessing prediction...")
    encoded_data = encode_game_data(game_data, encoders)
    
    results = predict_winner(lr_model, rf_model, encoded_data, game_data)
    
    display_results(game_data, results)
    
    print("\nGenerating prediction visualizations...")
    plot_prediction_graphs(game_data, results)
    
    print("\nPrediction complete!")
    
    another = input("\nPredict another game? (y/n): ").strip().lower()
    if another == 'y':
        main()

if __name__ == "__main__":
    main()