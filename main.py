import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# --- LANDetector CLASS ---
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

# Prepare features for LAN
X = df.drop(columns=["user"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run LAN anomaly detection
detector = LANDetector(k=5)
lan_scores = detector.fit_predict(X_scaled)
df['lan_score'] = lan_scores

# Mark top 5% as anomaly (pseudo-labels)
thresh = np.percentile(lan_scores, 95)
df['pseudo_label'] = (lan_scores > thresh).astype(int)

# Compute z-scores for features
feature_cols = [col for col in df.columns if col not in ['user', 'lan_score', 'pseudo_label']]
feature_means = df[feature_cols].mean()
feature_stds = df[feature_cols].std(ddof=0)
df_z = (df[feature_cols] - feature_means) / (feature_stds + 1e-8)

# Output all threats with details
suspicious = df[df['pseudo_label'] == 1].sort_values('lan_score', ascending=False)
logprint(f"Total users analyzed: {len(df)}")
logprint(f"Number of suspicious users detected: {len(suspicious)}")
logprint("\nAll Suspicious Users (LAN):")
logprint("=" * 80)
for idx, (i, user) in enumerate(suspicious.iterrows(), 1):
    rank = idx
    percentile = 100.0 * (len(df) - df['lan_score'].rank(method='min')[i]) / len(df)
    logprint(f"User: {user['user']}")
    logprint(f"  Rank: {rank} of {len(df)} (Anomaly Score Percentile: {percentile:.2f}%)")
    logprint(f"  LAN Anomaly Score: {user['lan_score']:.3f}")
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

# Precision-Recall and ROC curves
precision, recall, _ = precision_recall_curve(df['pseudo_label'], df['lan_score'])
fpr, tpr, _ = roc_curve(df['pseudo_label'], df['lan_score'])
roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (LAN)')
plt.legend()
plt.savefig('output/precision_recall_curve.png')
plt.close()

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (LAN)')
plt.legend()
plt.savefig('output/roc_curve.png')
plt.close()

logprint("Precision-Recall and ROC curves saved in 'output/' folder.")
