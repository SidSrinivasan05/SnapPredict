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
XGB_MODEL = os.path.join(OUT_DIR, "xgb_model.pkl")
TEST_DATA_FILE = os.path.join(OUT_DIR, "test_data.pkl")

def load_models_and_data():
    """Load trained models and test data"""
    with open(LOGISTIC_MODEL, "rb") as f:
        lr_model = pickle.load(f)
    
    with open(RF_MODEL, "rb") as f:
        rf_model = pickle.load(f)
    
    with open(XGB_MODEL, "rb") as f:
        xgb_model = pickle.load(f)
    
    with open(TEST_DATA_FILE, "rb") as f:
        test_data = pickle.load(f)
    
    return lr_model, rf_model, xgb_model, test_data

def plot_confusion_matrices(y_test, lr_pred, rf_pred, xgb_pred):
    """Create confusion matrix comparison for all three models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Logistic Regression
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    axes[0].set_title('Logistic Regression\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=12)
    
    # Random Forest
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    axes[1].set_title('Random Forest\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_xlabel('Predicted', fontsize=12)
    
    # XGBoost
    cm_xgb = confusion_matrix(y_test, xgb_pred)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    axes[2].set_title('XGBoost\nConfusion Matrix', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Actual', fontsize=12)
    axes[2].set_xlabel('Predicted', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "confusion_matrices.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_roc_curves(y_test, lr_proba, rf_proba, xgb_proba):
    """Create ROC curve comparison for all three models"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Logistic Regression ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba[:, 1])
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba[:, 1])
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba[:, 1])
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    # Plot
    ax.plot(fpr_lr, tpr_lr, color='blue', lw=2, 
            label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
    ax.plot(fpr_rf, tpr_rf, color='green', lw=2,
            label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
    ax.plot(fpr_xgb, tpr_xgb, color='orange', lw=2,
            label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
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
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_feature_importance_comparison(rf_model, xgb_model, feature_names):
    """Plot feature importance comparison between Random Forest and XGBoost"""
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_
    
    # Get top 15 features from XGBoost
    indices = np.argsort(xgb_importances)[::-1][:15]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Random Forest
    axes[0].barh(range(len(indices)), rf_importances[indices], color='forestgreen', alpha=0.8)
    axes[0].set_yticks(range(len(indices)))
    axes[0].set_yticklabels([feature_names[i] for i in indices])
    axes[0].set_xlabel('Feature Importance', fontsize=12)
    axes[0].set_title('Random Forest: Top 15 Feature Importances', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # XGBoost
    axes[1].barh(range(len(indices)), xgb_importances[indices], color='darkorange', alpha=0.8)
    axes[1].set_yticks(range(len(indices)))
    axes[1].set_yticklabels([feature_names[i] for i in indices])
    axes[1].set_xlabel('Feature Importance', fontsize=12)
    axes[1].set_title('XGBoost: Top 15 Feature Importances', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "feature_importance_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_precision_recall_curves(y_test, lr_proba, rf_proba, xgb_proba):
    """Create Precision-Recall curve comparison for all three models"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Logistic Regression
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_proba[:, 1])
    
    # Random Forest
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_proba[:, 1])
    
    # XGBoost
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_proba[:, 1])
    
    # Plot
    ax.plot(recall_lr, precision_lr, color='blue', lw=2, 
            label='Logistic Regression')
    ax.plot(recall_rf, precision_rf, color='green', lw=2,
            label='Random Forest')
    ax.plot(recall_xgb, precision_xgb, color='orange', lw=2,
            label='XGBoost')
    
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
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_model_comparison_metrics(y_test, lr_pred, rf_pred, xgb_pred, lr_proba, rf_proba, xgb_proba):
    """Create bar chart comparing key metrics across all three models"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': [
            accuracy_score(y_test, lr_pred),
            accuracy_score(y_test, rf_pred),
            accuracy_score(y_test, xgb_pred)
        ],
        'Precision': [
            precision_score(y_test, lr_pred),
            precision_score(y_test, rf_pred),
            precision_score(y_test, xgb_pred)
        ],
        'Recall': [
            recall_score(y_test, lr_pred),
            recall_score(y_test, rf_pred),
            recall_score(y_test, xgb_pred)
        ],
        'F1-Score': [
            f1_score(y_test, lr_pred),
            f1_score(y_test, rf_pred),
            f1_score(y_test, xgb_pred)
        ],
        'ROC-AUC': [
            auc(*roc_curve(y_test, lr_proba[:, 1])[:2]),
            auc(*roc_curve(y_test, rf_proba[:, 1])[:2]),
            auc(*roc_curve(y_test, xgb_proba[:, 1])[:2])
        ]
    }
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    lr_scores = [metrics[m][0] for m in metrics]
    rf_scores = [metrics[m][1] for m in metrics]
    xgb_scores = [metrics[m][2] for m in metrics]
    
    bars1 = ax.bar(x - width, lr_scores, width, label='Logistic Regression', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, rf_scores, width, label='Random Forest',
                   color='forestgreen', alpha=0.8)
    bars3 = ax.bar(x + width, xgb_scores, width, label='XGBoost',
                   color='darkorange', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (All Three Models)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_xgboost_learning_curve(xgb_model):
    """Plot XGBoost learning curve showing training progression"""
    results = xgb_model.evals_result()
    
    if results and 'validation_0' in results:
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, results['validation_0']['logloss'], label='Test')
        ax.set_xlabel('Boosting Round', fontsize=12)
        ax.set_ylabel('Log Loss', fontsize=12)
        ax.set_title('XGBoost Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(GRAPHS_DIR, "xgboost_learning_curve.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()

def plot_xgboost_tree_depth_analysis(y_test, xgb_proba):
    """Analyze XGBoost prediction confidence distribution"""
    home_probs = xgb_proba[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(home_probs[y_test == 1], label="Actual Home Wins", 
                 color="green", kde=True, stat="density", alpha=0.6, ax=ax)
    sns.histplot(home_probs[y_test == 0], label="Actual Away Wins", 
                 color="red", kde=True, stat="density", alpha=0.6, ax=ax)
    
    ax.set_title("XGBoost Predicted Probability Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Probability Home Team Wins", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(GRAPHS_DIR, "xgboost_probability_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

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
    print(f"✅ Saved: {output_path}")
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
    print(f"✅ Saved: {output_path}")
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
    print(f"✅ Saved: {output_path}")
    plt.close()

def create_model_analysis_summary(y_test, lr_pred, rf_pred, xgb_pred, lr_proba, rf_proba, xgb_proba):
    """Create comprehensive analysis summary table"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                  f1_score, log_loss, matthews_corrcoef)
    
    models = ['Logistic Regression', 'Random Forest', 'XGBoost']
    predictions = [lr_pred, rf_pred, xgb_pred]
    probabilities = [lr_proba, rf_proba, xgb_proba]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate metrics for each model
    metrics_data = []
    for model, pred, proba in zip(models, predictions, probabilities):
        metrics_data.append([
            model,
            f"{accuracy_score(y_test, pred):.4f}",
            f"{precision_score(y_test, pred):.4f}",
            f"{recall_score(y_test, pred):.4f}",
            f"{f1_score(y_test, pred):.4f}",
            f"{auc(*roc_curve(y_test, proba[:, 1])[:2]):.4f}",
            f"{log_loss(y_test, proba):.4f}",
            f"{matthews_corrcoef(y_test, pred):.4f}"
        ])
    
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 
               'ROC-AUC', 'Log Loss', 'MCC']
    
    table = ax.table(cellText=metrics_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows
    colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0']
    for i, color in enumerate(colors):
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor(color)
    
    plt.title('Comprehensive Model Performance Metrics', 
              fontsize=16, fontweight='bold', pad=20)
    
    output_path = os.path.join(GRAPHS_DIR, "model_metrics_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    print("Loading models and test data...")
    lr_model, rf_model, xgb_model, test_data = load_models_and_data()
    
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    feature_names = test_data["feature_names"]
    
    print("Generating predictions...")
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)
    
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    
    print("\nCreating visualizations...")
    
    # Logistic Regression specific plots
    plot_logistic_coefficients(lr_model, feature_names)
    plot_logistic_calibration_curve(y_test, lr_proba)
    plot_logistic_probability_distribution(y_test, lr_proba)
    
    # XGBoost specific plots
    plot_xgboost_learning_curve(xgb_model)
    plot_xgboost_tree_depth_analysis(y_test, xgb_proba)
    
    # Comparison plots (all three models)
    plot_confusion_matrices(y_test, lr_pred, rf_pred, xgb_pred)
    plot_roc_curves(y_test, lr_proba, rf_proba, xgb_proba)
    plot_feature_importance_comparison(rf_model, xgb_model, feature_names)
    plot_precision_recall_curves(y_test, lr_proba, rf_proba, xgb_proba)
    plot_model_comparison_metrics(y_test, lr_pred, rf_pred, xgb_pred, lr_proba, rf_proba, xgb_proba)
    create_model_analysis_summary(y_test, lr_pred, rf_pred, xgb_pred, lr_proba, rf_proba, xgb_proba)
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, lr_pred, target_names=['Away Win', 'Home Win']))
    
    print("\n--- Random Forest ---")
    print(classification_report(y_test, rf_pred, target_names=['Away Win', 'Home Win']))
    
    print("\n--- XGBoost ---")
    print(classification_report(y_test, xgb_pred, target_names=['Away Win', 'Home Win']))
    
    print(f"\n✅ All graphs saved to: {GRAPHS_DIR}")

if __name__ == "__main__":
    main()