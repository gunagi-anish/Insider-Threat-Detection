import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from models.advanced_anomaly_detector import AdvancedAnomalyDetector

# --- LANDetector CLASS (Legacy) ---
class LANDetector:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.scores_ = None

    def fit(self, X):
        nbrs = NearestNeighbors(n_neighbors=self.k+1, metric=self.metric)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        local_density = np.mean(distances[:, 1:], axis=1)
        self.scores_ = (local_density - np.min(local_density)) / (np.max(local_density) - np.min(local_density) + 1e-8)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.scores_

# --- OUTPUT FOLDER SETUP ---
os.makedirs('output', exist_ok=True)
log_lines = []
def logprint(*args, **kwargs):
    print(*args, **kwargs)
    log_lines.append(' '.join(str(a) for a in args))

# Load processed features
logprint("Loading features...")
data_path = "data/processed/features.csv"
df = pd.read_csv(data_path)

# Prepare features for advanced anomaly detection
X = df.drop(columns=["user"])

# Initialize and train advanced anomaly detector
logprint("Initializing Advanced Anomaly Detector...")
advanced_detector = AdvancedAnomalyDetector(
    device='cpu',  # Use 'cuda' if you have GPU
    autoencoder_params={
        'encoding_dim': 8,  # Reduced dimensionality
        'dropout_rate': 0.2
    }
)

logprint("Training Advanced Anomaly Detector...")
advanced_detector.fit(X)

# Get advanced anomaly scores
logprint("Computing advanced anomaly scores...")
advanced_scores = advanced_detector.predict(X)
df['advanced_score'] = advanced_scores

# Also run legacy LAN for comparison
logprint("Running legacy LAN detector for comparison...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
detector = LANDetector(k=5)
lan_scores = detector.fit_predict(X_scaled)
df['lan_score'] = lan_scores

# Mark top 5% as anomaly (pseudo-labels) using advanced scores
thresh = np.percentile(advanced_scores, 95)
df['pseudo_label'] = (advanced_scores > thresh).astype(int)

# Optimize ensemble weights if we have ground truth (optional)
# advanced_detector.optimize_weights(X, df['pseudo_label'])

# Compute z-scores for features
feature_cols = [col for col in df.columns if col not in ['user', 'lan_score', 'advanced_score', 'pseudo_label']]
feature_means = df[feature_cols].mean()
feature_stds = df[feature_cols].std(ddof=0)
df_z = (df[feature_cols] - feature_means) / (feature_stds + 1e-8)

# Output all threats with details (using advanced scores)
suspicious = df[df['pseudo_label'] == 1].sort_values('advanced_score', ascending=False)
logprint(f"Total users analyzed: {len(df)}")
logprint(f"Number of suspicious users detected: {len(suspicious)}")

# Performance comparison
lan_auc = roc_auc_score(df['pseudo_label'], df['lan_score'])
advanced_auc = roc_auc_score(df['pseudo_label'], df['advanced_score'])
logprint(f"\nPerformance Comparison:")
logprint(f"Legacy LAN AUC: {lan_auc:.4f}")
logprint(f"Advanced Model AUC: {advanced_auc:.4f}")
logprint(f"Improvement: {((advanced_auc - lan_auc) / lan_auc * 100):.2f}%")

logprint("\nAll Suspicious Users (Advanced Model):")
logprint("=" * 80)
for idx, (i, user) in enumerate(suspicious.iterrows(), 1):
    rank = idx
    percentile = 100.0 * (len(df) - df['advanced_score'].rank(method='min')[i]) / len(df)
    logprint(f"User: {user['user']}")
    logprint(f"  Rank: {rank} of {len(df)} (Anomaly Score Percentile: {percentile:.2f}%)")
    logprint(f"  Advanced Anomaly Score: {user['advanced_score']:.3f}")
    logprint(f"  Legacy LAN Score: {user['lan_score']:.3f}")
    logprint("  Feature values:")
    for col in feature_cols:
        logprint(f"    {col}: {user[col]}")
    logprint("  Feature z-scores:")
    for col in feature_cols:
        logprint(f"    {col}: {df_z.loc[i, col]:.2f}")
    logprint("-" * 80)

# Save results
suspicious.to_csv('output/lan_threat_detection_results.csv', index=False)
with open('output/output_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))

# Precision-Recall and ROC curves for both models
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Legacy LAN curves
precision_lan, recall_lan, _ = precision_recall_curve(df['pseudo_label'], df['lan_score'])
fpr_lan, tpr_lan, _ = roc_curve(df['pseudo_label'], df['lan_score'])
roc_auc_lan = auc(fpr_lan, tpr_lan)
pr_auc_lan = auc(recall_lan, precision_lan)

# Advanced model curves
precision_adv, recall_adv, _ = precision_recall_curve(df['pseudo_label'], df['advanced_score'])
fpr_adv, tpr_adv, _ = roc_curve(df['pseudo_label'], df['advanced_score'])
roc_auc_adv = auc(fpr_adv, tpr_adv)
pr_auc_adv = auc(recall_adv, precision_adv)

# Plot Precision-Recall curves
axes[0, 0].plot(recall_lan, precision_lan, label=f'Legacy LAN (AUC = {pr_auc_lan:.3f})', color='blue')
axes[0, 0].plot(recall_adv, precision_adv, label=f'Advanced Model (AUC = {pr_auc_adv:.3f})', color='red')
axes[0, 0].set_xlabel('Recall')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].set_title('Precision-Recall Curve Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot ROC curves
axes[0, 1].plot(fpr_lan, tpr_lan, label=f'Legacy LAN (AUC = {roc_auc_lan:.3f})', color='blue')
axes[0, 1].plot(fpr_adv, tpr_adv, label=f'Advanced Model (AUC = {roc_auc_adv:.3f})', color='red')
axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Score distribution comparison
axes[1, 0].hist(df[df['pseudo_label'] == 0]['lan_score'], bins=30, alpha=0.7, label='Normal (LAN)', color='blue')
axes[1, 0].hist(df[df['pseudo_label'] == 1]['lan_score'], bins=30, alpha=0.7, label='Anomaly (LAN)', color='red')
axes[1, 0].set_xlabel('LAN Anomaly Score')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Legacy LAN Score Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].hist(df[df['pseudo_label'] == 0]['advanced_score'], bins=30, alpha=0.7, label='Normal (Advanced)', color='blue')
axes[1, 1].hist(df[df['pseudo_label'] == 1]['advanced_score'], bins=30, alpha=0.7, label='Anomaly (Advanced)', color='red')
axes[1, 1].set_xlabel('Advanced Anomaly Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Advanced Model Score Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('output/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Save individual curves as well
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall_adv, precision_adv, label=f'Advanced Model (AUC = {pr_auc_adv:.3f})', color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Advanced Model)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(fpr_adv, tpr_adv, label=f'Advanced Model (AUC = {roc_auc_adv:.3f})', color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Advanced Model)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('output/advanced_model_curves.png', dpi=300, bbox_inches='tight')
plt.close()

logprint("Model comparison and evaluation curves saved in 'output/' folder.")
