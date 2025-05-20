import pandas as pd
import numpy as np
from preprocessing.extract_features import extract_features

def process_chunk(chunk, activity_type):
    """Process a chunk of data and return aggregated features"""
    chunk['logon'] = 1 if activity_type == 'logon' else 0
    chunk['file'] = 1 if activity_type == 'file' else 0
    chunk['email'] = 1 if activity_type == 'email' else 0
    chunk['device'] = 1 if activity_type == 'device' else 0
    chunk['http'] = 1 if activity_type == 'http' else 0
    return chunk[['user', 'date', 'logon', 'file', 'email', 'device', 'http']]

def process_file(file_path, activity_type, chunk_size=100000):
    """Process a file in chunks and return aggregated features"""
    print(f"Processing {activity_type} data...")
    
    # Initialize empty DataFrame for results
    all_features = pd.DataFrame()
    
    # Process file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Convert date column to datetime
        chunk['date'] = pd.to_datetime(chunk['date']).dt.date
        
        # Process chunk
        processed_chunk = process_chunk(chunk, activity_type)
        
        # Aggregate features for this chunk
        chunk_features = extract_features(processed_chunk)
        
        # Combine with previous results
        if all_features.empty:
            all_features = chunk_features
        else:
            # Merge with existing features, summing the activity counts
            all_features = pd.merge(all_features, chunk_features, on='user', how='outer', suffixes=('', '_new'))
            # Sum the numeric columns
            for col in ['logon', 'file', 'email', 'device', 'http']:
                all_features[col] = all_features[col].fillna(0) + all_features[f'{col}_new'].fillna(0)
                all_features = all_features.drop(columns=[f'{col}_new'])
    
    return all_features

# Process each file
print("Starting feature extraction...")

# Process each file separately and combine results
logon_features = process_file('data/raw/logon.csv', 'logon')
file_features = process_file('data/raw/file.csv', 'file')
email_features = process_file('data/raw/email.csv', 'email')
device_features = process_file('data/raw/device.csv', 'device')
http_features = process_file('data/raw/http.csv', 'http')

# Combine all features
print("Combining features from all sources...")
all_features = pd.concat([logon_features, file_features, email_features, device_features, http_features])
final_features = all_features.groupby('user').sum().reset_index()

# Save to processed folder
print("Saving features...")
final_features.to_csv('data/processed/features.csv', index=False)
print("Features extracted and saved to data/processed/features.csv")
