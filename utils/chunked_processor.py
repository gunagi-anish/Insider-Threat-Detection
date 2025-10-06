import pandas as pd
import numpy as np
import os
import gc
from typing import Generator, Tuple, Optional
import psutil
import time
from .date_parser import FlexibleDateParser

class ChunkedProcessor:
    """Utility class for processing large CSV files in chunks to manage memory"""
    
    def __init__(self, chunk_size: int = 10000, max_memory_usage: float = 0.8):
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        self.processed_rows = 0
        self.total_rows = 0
        self.date_parser = FlexibleDateParser()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    def estimate_file_size(self, file_path: str) -> int:
        """Estimate number of rows in CSV file"""
        try:
            # Read first few lines to estimate
            sample_df = pd.read_csv(file_path, nrows=1000)
            file_size = os.path.getsize(file_path)
            estimated_rows = int((file_size / len(sample_df.to_csv().encode())) * len(sample_df))
            return estimated_rows
        except Exception:
            return 0
    
    def read_csv_chunks(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Read CSV file in chunks"""
        try:
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)
            for chunk in chunk_iter:
                # Check memory usage
                if self.get_memory_usage() > self.max_memory_usage:
                    gc.collect()  # Force garbage collection
                    time.sleep(0.1)  # Brief pause
                
                yield chunk
        except Exception as e:
            raise Exception(f"Error reading CSV chunks: {str(e)}")
    
    def process_large_file(self, file_path: str, processing_func) -> pd.DataFrame:
        """Process large CSV file in chunks"""
        print(f"Processing large file: {file_path}")
        
        # Estimate total rows
        self.total_rows = self.estimate_file_size(file_path)
        print(f"Estimated total rows: {self.total_rows:,}")
        
        # Initialize result storage
        results = []
        self.processed_rows = 0
        
        try:
            # Process in chunks
            for chunk in self.read_csv_chunks(file_path):
                # Process chunk
                processed_chunk = processing_func(chunk)
                results.append(processed_chunk)
                
                self.processed_rows += len(chunk)
                progress = (self.processed_rows / self.total_rows) * 100 if self.total_rows > 0 else 0
                
                print(f"Processed {self.processed_rows:,} rows ({progress:.1f}%)")
                
                # Memory management
                if len(results) > 10:  # Keep only last 10 chunks in memory
                    # Combine and save intermediate results
                    combined = pd.concat(results, ignore_index=True)
                    results = [combined]
                    gc.collect()
            
            # Combine all results
            if results:
                final_result = pd.concat(results, ignore_index=True)
                print(f"Processing complete. Final dataset: {len(final_result):,} rows")
                return final_result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error processing large file: {str(e)}")
            raise
    
    def extract_features_chunked(self, file_path: str) -> pd.DataFrame:
        """Extract features from large CSV file in chunks"""
        def process_chunk(chunk):
            # Basic feature extraction for chunk
            if 'user' in chunk.columns and 'date' in chunk.columns:
                # Parse dates flexibly
                chunk['date'] = self.date_parser.parse_date_column(chunk['date'])
                
                # Remove rows with invalid dates (skip problematic data)
                original_count = len(chunk)
                chunk = chunk.dropna(subset=['date'])
                skipped_count = original_count - len(chunk)
                
                if skipped_count > 0:
                    print(f"Info: Skipped {skipped_count} rows with invalid dates in chunk")
                
                if len(chunk) == 0:
                    return pd.DataFrame()
                
                # Group by user and date, then aggregate
                numeric_cols = [col for col in chunk.columns if col not in ['user', 'date']]
                
                # Ensure we have the expected activity columns, add defaults if missing
                expected_cols = ['logon', 'file', 'email', 'device', 'http']
                for col in expected_cols:
                    if col not in chunk.columns:
                        chunk[col] = 0
                        numeric_cols.append(col)
                
                if numeric_cols:
                    features = chunk.groupby(['user', 'date'])[numeric_cols].sum().reset_index()
                    return features
            return chunk
        
        return self.process_large_file(file_path, process_chunk)
    
    def merge_large_files(self, file_paths: dict) -> pd.DataFrame:
        """Merge large CSV files efficiently"""
        print("Merging large files...")
        
        # First pass: collect all unique users and dates
        all_users = set()
        all_dates = set()
        
        for file_type, file_path in file_paths.items():
            if file_path and os.path.exists(file_path):
                print(f"Scanning {file_type} file for users and dates...")
                for chunk in self.read_csv_chunks(file_path):
                    if 'user' in chunk.columns:
                        all_users.update(chunk['user'].unique())
                    if 'date' in chunk.columns:
                        all_dates.update(pd.to_datetime(chunk['date']).dt.date)
        
        print(f"Found {len(all_users)} unique users and {len(all_dates)} unique dates")
        
        # Create base dataframe
        base_data = []
        for user in all_users:
            for date in all_dates:
                base_data.append({
                    'user': user,
                    'date': date,
                    'logon': 0,
                    'file': 0,
                    'email': 0,
                    'device': 0,
                    'http': 0
                })
        
        merged_data = pd.DataFrame(base_data)
        merged_data['date'] = pd.to_datetime(merged_data['date'])
        
        # Second pass: merge data from each file
        for file_type, file_path in file_paths.items():
            if file_path and os.path.exists(file_path):
                print(f"Merging {file_type} data...")
                for chunk in self.read_csv_chunks(file_path):
                    chunk['date'] = pd.to_datetime(chunk['date'])
                    
                    for _, row in chunk.iterrows():
                        mask = (merged_data['user'] == row['user']) & (merged_data['date'].dt.date == row['date'].date())
                        if mask.any():
                            merged_data.loc[mask, file_type] = row[file_type]
        
        # Convert date back to string
        merged_data['date'] = merged_data['date'].dt.strftime('%Y-%m-%d')
        
        print(f"Merge complete. Final dataset: {len(merged_data):,} rows")
        return merged_data
