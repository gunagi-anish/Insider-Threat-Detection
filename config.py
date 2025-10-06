"""
Configuration settings for Insider Threat Detection System
"""

import os

class Config:
    """Base configuration class"""
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024 * 1024  # 50GB
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Large file processing settings
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
    CHUNK_SIZE = 10000  # Rows per chunk
    MAX_MEMORY_USAGE = 0.8  # 80% of available memory
    
    # Model settings
    AUTOENCODER_EPOCHS = 100
    AUTOENCODER_BATCH_SIZE = 32
    AUTOENCODER_LEARNING_RATE = 0.001
    AUTOENCODER_ENCODING_DIM = 8
    AUTOENCODER_DROPOUT_RATE = 0.2
    
    # Anomaly detection settings
    ANOMALY_THRESHOLD_PERCENTILE = 95  # Top 5% as anomalies
    MIN_SAMPLES_FOR_TRAINING = 100
    
    # Visualization settings
    MAX_POINTS_FOR_PLOTS = 10000  # Limit points for performance
    PLOT_DPI = 300
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    
    # Performance settings
    ENABLE_CACHING = False  # Disable for large files
    SEND_FILE_MAX_AGE_DEFAULT = 0
    
    @staticmethod
    def get_file_size_mb(file_path):
        """Get file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    @staticmethod
    def get_file_size_gb(file_path):
        """Get file size in GB"""
        return os.path.getsize(file_path) / (1024 * 1024 * 1024)
    
    @staticmethod
    def is_large_file(file_path):
        """Check if file is considered large"""
        return Config.get_file_size_mb(file_path) > (Config.LARGE_FILE_THRESHOLD / (1024 * 1024))
