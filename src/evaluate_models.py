# src/evaluate_model.py
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report
)

OUT_DIR = "output/models"
GRAPHS_DIR = "output/graphs"
LOGISTIC_MODEL = os.path.join(OUT_DIR, "logistic_model.pkl")
RF_MODEL = os.path.join(OUT_DIR, "rf_model.pkl")
TEST_DATA_FILE = os.path.join(OUT_DIR, "test_data.pkl")

def load_models_and_data():
    """Load trained models and test data"""
    with open(LOGISTIC_MODEL, "rb") as f:
        lr_model = pickle.load(f)
    
    with open(RF_MODEL, "rb") as f:
        rf_model = pickle.load(f)
    
    with open(TEST_DATA_FILE, "rb") as f:
        test_data = pickle.load(f)
    
    return lr_model, rf_model, test_data

def plot_confusion_matrices(y_test, lr_pred, rf_pred):
    """Create confusion matrix comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    axes[0].set_title('Logistic Regression\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=12)
    
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    axes[1].set_title('Random Forest\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_xlabel('Predicted', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "confusion_matrices.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_roc_curves(y_test, lr_proba, rf_proba):
    """Create ROC curve comparison"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba[:, 1])
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba[:, 1])
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    ax.plot(fpr_lr, tpr_lr, color='blue', lw=2, 
            label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
    ax.plot(fpr_rf, tpr_rf, color='green', lw=2,
            label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
            label='Random Guess (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "roc_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_feature_importance(rf_model, feature_names):
    """Plot Random Forest feature importance"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(indices)), importances[indices], color='forestgreen', alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Random Forest: Top 15 Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_precision_recall_curves(y_test, lr_proba, rf_proba):
    """Create Precision-Recall curve comparison"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_proba[:, 1])
    
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_proba[:, 1])
    
    ax.plot(recall_lr, precision_lr, color='blue', lw=2, 
            label='Logistic Regression')
    ax.plot(recall_rf, precision_rf, color='green', lw=2,
            label='Random Forest')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves: Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "precision_recall_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_model_comparison_metrics(y_test, lr_pred, rf_pred, lr_proba, rf_proba):
    """Create bar chart comparing key metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': [
            accuracy_score(y_test, lr_pred),
            accuracy_score(y_test, rf_pred)
        ],
        'Precision': [
            precision_score(y_test, lr_pred),
            precision_score(y_test, rf_pred)
        ],
        'Recall': [
            recall_score(y_test, lr_pred),
            recall_score(y_test, rf_pred)
        ],
        'F1-Score': [
            f1_score(y_test, lr_pred),
            f1_score(y_test, rf_pred)
        ],
        'ROC-AUC': [
            auc(*roc_curve(y_test, lr_proba[:, 1])[:2]),
            auc(*roc_curve(y_test, rf_proba[:, 1])[:2])
        ]
    }
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    lr_scores = [metrics[m][0] for m in metrics]
    rf_scores = [metrics[m][1] for m in metrics]
    
    bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest',
                   color='forestgreen', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    print("Loading models and test data...")
    lr_model, rf_model, test_data = load_models_and_data()
    
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    feature_names = test_data["feature_names"]
    
    print("Generating predictions...")
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)
    
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    
    print("\nCreating visualizations...")
    
    plot_logistic_coefficients(lr_model, feature_names)
    plot_logistic_calibration_curve(y_test, lr_proba)
    plot_logistic_probability_distribution(y_test, lr_proba)
    
    plot_confusion_matrices(y_test, lr_pred, rf_pred)
    plot_roc_curves(y_test, lr_proba, rf_proba)
    plot_feature_importance(rf_model, feature_names)
    plot_precision_recall_curves(y_test, lr_proba, rf_proba)
    plot_model_comparison_metrics(y_test, lr_pred, rf_pred, lr_proba, rf_proba)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, lr_pred, target_names=['Away Win', 'Home Win']))
    
    print("\n--- Random Forest ---")
    print(classification_report(y_test, rf_pred, target_names=['Away Win', 'Home Win']))
    
    print(f"\nAll graphs saved to: {GRAPHS_DIR}")

def plot_logistic_coefficients(lr_model, feature_names):
    """Plot coefficient weights for Logistic Regression"""
    coef = lr_model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef
    }).sort_values(by='Coefficient', ascending=False)

    top_coeff = pd.concat([coef_df.head(12), coef_df.tail(12)])

    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_coeff, x='Coefficient', y='Feature', palette='coolwarm')
    plt.title("Logistic Regression: Top Positive & Negative Coefficients", fontsize=14, fontweight='bold')
    plt.xlabel("Coefficient Weight")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path = os.path.join(GRAPHS_DIR, "logistic_coefficients.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_logistic_calibration_curve(y_test, lr_proba):
    """Plot calibration curve for probability reliability"""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_test, lr_proba[:, 1], n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label="Logistic Regression")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Win Frequency")
    plt.title("Calibration Curve: Logistic Regression", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = os.path.join(GRAPHS_DIR, "logistic_calibration_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_logistic_probability_distribution(y_test, lr_proba):
    """Plot distribution of predicted win probabilities"""
    home_probs = lr_proba[:, 1]

    plt.figure(figsize=(10, 6))
    sns.histplot(home_probs[y_test == 1], label="Home Wins", color="green", kde=True, stat="density", alpha=0.6)
    sns.histplot(home_probs[y_test == 0], label="Away Wins", color="red", kde=True, stat="density", alpha=0.6)

    plt.title("Logistic Regression Predicted Probability Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Probability Home Team Wins")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(GRAPHS_DIR, "logistic_probability_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()