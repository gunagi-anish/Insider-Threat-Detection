import pandas as pd

def extract_features(log_df):
    # Example aggregation of user actions per day
    features = log_df.groupby(['user', 'date']).agg({
        'logon': 'sum',
        'file': 'sum',
        'email': 'sum',
        'device': 'sum',
        'http': 'sum'
    }).reset_index()
    
    # Only average numeric columns, exclude 'date'
    numeric_cols = ['logon', 'file', 'email', 'device', 'http']
    user_features = features.groupby('user')[numeric_cols].mean().reset_index()
    return user_features
