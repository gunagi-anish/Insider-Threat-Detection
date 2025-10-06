import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class AutoencoderAnomalyDetector(nn.Module):
    """Advanced Autoencoder for Anomaly Detection"""
    
    def __init__(self, input_dim, encoding_dim=None, dropout_rate=0.2):
        super(AutoencoderAnomalyDetector, self).__init__()
        
        if encoding_dim is None:
            encoding_dim = max(2, input_dim // 4)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class AdvancedAnomalyDetector:
    """Advanced Ensemble Anomaly Detection System"""
    
    def __init__(self, device='cpu', autoencoder_params=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.models = {}
        self.weights = {}
        self.autoencoder_params = autoencoder_params or {}
        
    def _create_advanced_features(self, df):
        """Create advanced features from basic behavioral data"""
        feature_cols = [col for col in df.columns if col not in ['user', 'lan_score', 'pseudo_label']]
        X = df[feature_cols].copy()
        
        # 1. Statistical features
        X['total_activity'] = X.sum(axis=1)
        X['activity_std'] = X[feature_cols].std(axis=1)
        X['activity_mean'] = X[feature_cols].mean(axis=1)
        X['activity_skew'] = X[feature_cols].skew(axis=1)
        X['activity_kurtosis'] = X[feature_cols].kurtosis(axis=1)
        
        # 2. Ratio features
        X['logon_ratio'] = X['logon'] / (X['total_activity'] + 1e-8)
        X['file_ratio'] = X['file'] / (X['total_activity'] + 1e-8)
        X['email_ratio'] = X['email'] / (X['total_activity'] + 1e-8)
        X['device_ratio'] = X['device'] / (X['total_activity'] + 1e-8)
        X['http_ratio'] = X['http'] / (X['total_activity'] + 1e-8)
        
        # 3. Interaction features
        X['logon_file_interaction'] = X['logon'] * X['file']
        X['email_http_interaction'] = X['email'] * X['http']
        X['device_activity_interaction'] = X['device'] * X['total_activity']
        
        # 4. Normalized features (z-scores)
        for col in feature_cols:
            X[f'{col}_normalized'] = (X[col] - X[col].mean()) / (X[col].std() + 1e-8)
        
        # 5. Percentile features
        for col in feature_cols:
            X[f'{col}_percentile'] = X[col].rank(pct=True)
        
        # 6. Polynomial features (degree 2)
        for col in feature_cols:
            X[f'{col}_squared'] = X[col] ** 2
        
        return X
    
    def _train_autoencoder(self, X_train, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the autoencoder model"""
        input_dim = X_train.shape[1]
        
        # Initialize model
        model = AutoencoderAnomalyDetector(
            input_dim=input_dim,
            encoding_dim=self.autoencoder_params.get('encoding_dim'),
            dropout_rate=self.autoencoder_params.get('dropout_rate', 0.2)
        ).to(self.device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in dataloader:
                data = batch[0]
                optimizer.zero_grad()
                
                reconstructed, encoded = model(data)
                loss = criterion(reconstructed, data)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return model, train_losses
    
    def fit(self, X, y=None):
        """Fit the ensemble of anomaly detection models"""
        # Create advanced features
        X_advanced = self._create_advanced_features(pd.DataFrame(X))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_advanced)
        
        # Train Autoencoder
        print("Training Autoencoder...")
        autoencoder, losses = self._train_autoencoder(X_scaled)
        self.models['autoencoder'] = autoencoder
        
        # Train other models
        print("Training Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.05, 
            random_state=42,
            n_estimators=200
        ).fit(X_scaled)
        
        print("Training One-Class SVM...")
        self.models['one_class_svm'] = OneClassSVM(
            nu=0.05,
            kernel='rbf',
            gamma='scale'
        ).fit(X_scaled)
        
        print("Training Local Outlier Factor...")
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True
        ).fit(X_scaled)
        
        # Initialize weights (will be optimized based on performance)
        self.weights = {
            'autoencoder': 0.4,
            'isolation_forest': 0.25,
            'one_class_svm': 0.2,
            'lof': 0.15
        }
        
        return self
    
    def predict_scores(self, X):
        """Get anomaly scores from all models"""
        # Create advanced features
        X_advanced = self._create_advanced_features(pd.DataFrame(X))
        X_scaled = self.scaler.transform(X_advanced)
        
        scores = {}
        
        # Autoencoder scores (reconstruction error)
        self.models['autoencoder'].eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            reconstructed, _ = self.models['autoencoder'](X_tensor)
            reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            scores['autoencoder'] = reconstruction_error.cpu().numpy()
        
        # Isolation Forest scores
        scores['isolation_forest'] = -self.models['isolation_forest'].decision_function(X_scaled)
        
        # One-Class SVM scores
        scores['one_class_svm'] = -self.models['one_class_svm'].decision_function(X_scaled)
        
        # LOF scores
        scores['lof'] = -self.models['lof'].decision_function(X_scaled)
        
        return scores
    
    def predict(self, X):
        """Get ensemble anomaly scores"""
        scores = self.predict_scores(X)
        
        # Normalize scores to [0, 1] range
        normalized_scores = {}
        for model_name, score in scores.items():
            score_min, score_max = score.min(), score.max()
            if score_max > score_min:
                normalized_scores[model_name] = (score - score_min) / (score_max - score_min)
            else:
                normalized_scores[model_name] = np.zeros_like(score)
        
        # Weighted ensemble
        ensemble_score = np.zeros(len(X))
        for model_name, weight in self.weights.items():
            ensemble_score += weight * normalized_scores[model_name]
        
        return ensemble_score
    
    def optimize_weights(self, X, y):
        """Optimize ensemble weights based on validation performance"""
        scores = self.predict_scores(X)
        
        # Normalize scores
        normalized_scores = {}
        for model_name, score in scores.items():
            score_min, score_max = score.min(), score.max()
            if score_max > score_min:
                normalized_scores[model_name] = (score - score_min) / (score_max - score_min)
            else:
                normalized_scores[model_name] = np.zeros_like(score)
        
        # Simple grid search for optimal weights
        best_auc = 0
        best_weights = self.weights.copy()
        
        # Test different weight combinations
        weight_candidates = [
            {'autoencoder': 0.5, 'isolation_forest': 0.3, 'one_class_svm': 0.1, 'lof': 0.1},
            {'autoencoder': 0.4, 'isolation_forest': 0.25, 'one_class_svm': 0.2, 'lof': 0.15},
            {'autoencoder': 0.3, 'isolation_forest': 0.4, 'one_class_svm': 0.2, 'lof': 0.1},
            {'autoencoder': 0.6, 'isolation_forest': 0.2, 'one_class_svm': 0.1, 'lof': 0.1},
        ]
        
        for weights in weight_candidates:
            ensemble_score = np.zeros(len(X))
            for model_name, weight in weights.items():
                ensemble_score += weight * normalized_scores[model_name]
            
            try:
                auc = roc_auc_score(y, ensemble_score)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = weights.copy()
            except:
                continue
        
        self.weights = best_weights
        print(f"Optimized weights: {self.weights}")
        print(f"Best AUC: {best_auc:.4f}")
        
        return best_weights
