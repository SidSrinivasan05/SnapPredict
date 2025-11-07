import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Load saved data
with open("output/models/test_data.pkl", "rb") as f:
    X_test, y_test = pickle.load(f)

with open("output/models/logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

# Predict
y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]

# ===== Confusion Matrix =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Away Win", "Home Win"], yticklabels=["Away Win", "Home Win"])
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/graphs/logistic_confusion_matrix.png")
plt.close()

# ===== ROC Curve =====
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1], linestyle="--")
plt.title("Logistic Regression - ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("output/graphs/logistic_roc_curve.png")
plt.close()

# ===== Precision-Recall Curve =====
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.title("Logistic Regression - Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("output/graphs/logistic_precision_recall.png")
plt.close()

# ===== Classification Report (Print Only) =====
print("\n=== Logistic Regression Report ===\n")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc:.3f}")
print("\n✅ Saved logistic-only graphs in output/graphs\n")