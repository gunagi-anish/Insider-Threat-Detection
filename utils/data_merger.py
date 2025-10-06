import pandas as pd
import numpy as np
from datetime import datetime
import os
from .chunked_processor import ChunkedProcessor
from .date_parser import FlexibleDateParser

class DataMerger:
    """Utility class to merge separate CSV files into a unified dataset"""
    
    def __init__(self):
        self.required_files = {
            'logon': ['user', 'date', 'logon'],
            'file': ['user', 'date', 'file'],
            'email': ['user', 'date', 'email'],
            'device': ['user', 'date', 'device'],
            'http': ['user', 'date', 'http']
        }
        self.chunked_processor = ChunkedProcessor()
        self.date_parser = FlexibleDateParser()
    
    def validate_file(self, file_path, file_type):
        """Validate that a CSV file has the required columns (flexible validation)"""
        try:
            # For large files, only read first chunk for validation
            df = pd.read_csv(file_path, nrows=1000)
            required_cols = self.required_files[file_type]
            
            # Check for essential columns (user and date are mandatory)
            essential_cols = ['user', 'date']
            missing_essential = [col for col in essential_cols if col not in df.columns]
            if missing_essential:
                return False, f"Missing essential columns in {file_type} file: {', '.join(missing_essential)}"
            
            # Check if date column can be parsed using flexible parser
            is_valid, message = self.date_parser.validate_date_column(df['date'])
            if not is_valid:
                print(f"Warning: Date validation failed in {file_type} file: {message}")
                print(f"Will attempt to process file anyway, skipping problematic rows")
                # Don't return False, continue with processing
            
            # Check for activity column (flexible - can be missing)
            activity_col = file_type
            if activity_col not in df.columns:
                print(f"Warning: {activity_col} column missing in {file_type} file. Will use default value 0.")
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error reading {file_type} file: {str(e)}"
    
    def merge_files(self, file_paths):
        """
        Merge separate CSV files into a unified dataset
        
        Args:
            file_paths: Dictionary with keys as file types and values as file paths
                       e.g., {'logon': 'path/to/logon.csv', 'file': 'path/to/file.csv', ...}
        
        Returns:
            tuple: (merged_dataframe, error_message)
        """
        try:
            # Check if any file is large (>100MB)
            large_files = []
            for file_type, file_path in file_paths.items():
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 100 * 1024 * 1024:  # 100MB
                        large_files.append(file_type)
            
            if large_files:
                print(f"Large files detected: {large_files}. Using chunked processing...")
                # Use chunked processing for large files
                merged_data = self.chunked_processor.merge_large_files(file_paths)
                return merged_data, None
            else:
                # Use regular processing for small files
                return self._merge_files_regular(file_paths)
            
        except Exception as e:
            return None, f"Error merging files: {str(e)}"
    
    def _merge_files_regular(self, file_paths):
        """Regular file merging for small files"""
        try:
            merged_data = None
            all_users = set()
            all_dates = set()
            
            # First pass: collect all users and dates
            for file_type, file_path in file_paths.items():
                if file_path and os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    all_users.update(df['user'].unique())
                    
                    # Parse dates flexibly
                    parsed_dates = self.date_parser.parse_date_column(df['date'])
                    valid_dates = parsed_dates.dropna()
                    if len(valid_dates) > 0:
                        all_dates.update(valid_dates.dt.date)
            
            # Create base dataframe with all user-date combinations
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
                    df = pd.read_csv(file_path)
                    
                    # Parse dates flexibly
                    df['date'] = self.date_parser.parse_date_column(df['date'])
                    
                    # Remove rows with invalid dates (skip problematic data)
                    original_count = len(df)
                    df = df.dropna(subset=['date'])
                    skipped_count = original_count - len(df)
                    
                    if skipped_count > 0:
                        print(f"Info: Skipped {skipped_count} rows with invalid dates in {file_path}")
                    
                    if len(df) == 0:
                        print(f"Warning: No valid dates found in {file_path}, skipping entire file")
                        continue
                    
                    # Check if activity column exists, if not use default value
                    if file_type not in df.columns:
                        print(f"Warning: {file_type} column not found in {file_path}, using default value 0")
                        df[file_type] = 0
                    
                    # Merge with base data
                    for _, row in df.iterrows():
                        if pd.notna(row['date']):
                            mask = (merged_data['user'] == row['user']) & (merged_data['date'].dt.date == row['date'].date())
                            if mask.any():
                                merged_data.loc[mask, file_type] = row[file_type]
            
            # Convert date back to string for consistency
            merged_data['date'] = merged_data['date'].dt.strftime('%Y-%m-%d')
            
            return merged_data, None
            
        except Exception as e:
            return None, f"Error merging files: {str(e)}"
    
    def create_sample_files(self, output_dir='sample_data'):
        """Create sample separate CSV files for testing"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample data
        users = ['AAE0190', 'AAF0535', 'AAF0791', 'AAL0706', 'AAM0658']
        dates = ['2023-01-01', '2023-01-02', '2023-01-03']
        
        # Logon data
        logon_data = []
        for user in users:
            for date in dates:
                logon_data.append({
                    'user': user,
                    'date': date,
                    'logon': np.random.randint(5, 25)
                })
        
        pd.DataFrame(logon_data).to_csv(f'{output_dir}/logon.csv', index=False)
        
        # File data
        file_data = []
        for user in users:
            for date in dates:
                file_data.append({
                    'user': user,
                    'date': date,
                    'file': np.random.randint(0, 20)
                })
        
        pd.DataFrame(file_data).to_csv(f'{output_dir}/file.csv', index=False)
        
        # Email data
        email_data = []
        for user in users:
            for date in dates:
                email_data.append({
                    'user': user,
                    'date': date,
                    'email': np.random.randint(10, 100)
                })
        
        pd.DataFrame(email_data).to_csv(f'{output_dir}/email.csv', index=False)
        
        # Device data
        device_data = []
        for user in users:
            for date in dates:
                device_data.append({
                    'user': user,
                    'date': date,
                    'device': np.random.randint(0, 15)
                })
        
        pd.DataFrame(device_data).to_csv(f'{output_dir}/device.csv', index=False)
        
        # HTTP data
        http_data = []
        for user in users:
            for date in dates:
                http_data.append({
                    'user': user,
                    'date': date,
                    'http': np.random.randint(1000, 10000)
                })
        
        pd.DataFrame(http_data).to_csv(f'{output_dir}/http.csv', index=False)
        
        return f"Sample files created in {output_dir}/ directory"


