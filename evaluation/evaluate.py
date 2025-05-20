from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(scores, labels):
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()
