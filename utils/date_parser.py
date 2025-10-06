import pandas as pd
import numpy as np
from datetime import datetime
import re

class FlexibleDateParser:
    """Utility class for parsing various date formats flexibly"""
    
    def __init__(self):
        # Common date formats to try
        self.date_formats = [
            '%Y-%m-%d',           # 2023-01-01
            '%Y/%m/%d',           # 2023/01/01
            '%m/%d/%Y',           # 01/01/2023
            '%d/%m/%Y',           # 01/01/2023
            '%m-%d-%Y',           # 01-01-2023
            '%d-%m-%Y',           # 01-01-2023
            '%Y-%m-%d %H:%M:%S',  # 2023-01-01 12:00:00
            '%Y/%m/%d %H:%M:%S',  # 2023/01/01 12:00:00
            '%m/%d/%Y %H:%M:%S',  # 01/01/2023 12:00:00
            '%d/%m/%Y %H:%M:%S',  # 01/01/2023 12:00:00
            '%Y-%m-%d %H:%M',     # 2023-01-01 12:00
            '%Y/%m/%d %H:%M',     # 2023/01/01 12:00
            '%m/%d/%Y %H:%M',     # 01/01/2023 12:00
            '%d/%m/%Y %H:%M',     # 01/01/2023 12:00
            '%Y%m%d',             # 20230101
            '%d-%b-%Y',           # 01-Jan-2023
            '%d-%B-%Y',           # 01-January-2023
            '%b %d, %Y',          # Jan 01, 2023
            '%B %d, %Y',          # January 01, 2023
        ]
    
    def parse_date_column(self, date_series):
        """Parse a pandas Series containing dates in various formats - skip problematic data"""
        if date_series.empty:
            return pd.Series([], dtype='datetime64[ns]')
        
        # First, handle ambiguous dates by expanding them
        expanded_series = self._expand_ambiguous_dates(date_series)
        
        # Try multiple parsing strategies with error skipping
        parsing_strategies = [
            ("mixed format inference (dayfirst=False)", lambda x: pd.to_datetime(x, format='mixed', errors='coerce', dayfirst=False)),
            ("mixed format inference (dayfirst=True)", lambda x: pd.to_datetime(x, format='mixed', errors='coerce', dayfirst=True)),
            ("ISO8601 format", lambda x: pd.to_datetime(x, format='ISO8601', errors='coerce')),
        ]
        
        # Add individual format strategies
        for date_format in self.date_formats:
            parsing_strategies.append((f"format {date_format}", lambda x, fmt=date_format: pd.to_datetime(x, format=fmt, errors='coerce')))
        
        # Try each strategy
        for strategy_name, strategy_func in parsing_strategies:
            try:
                parsed = strategy_func(expanded_series)
                valid_count = parsed.notna().sum()
                if valid_count > 0:
                    print(f"Successfully parsed {valid_count}/{len(expanded_series)} dates using {strategy_name}")
                    return parsed
            except Exception as e:
                print(f"Skipping {strategy_name} due to error: {str(e)}")
                continue
        
        # Try with cleaning
        try:
            cleaned_dates = self._clean_date_strings(expanded_series)
            for strategy_name, strategy_func in parsing_strategies[:3]:  # Try first 3 strategies with cleaned data
                try:
                    parsed = strategy_func(cleaned_dates)
                    valid_count = parsed.notna().sum()
                    if valid_count > 0:
                        print(f"Successfully parsed {valid_count}/{len(cleaned_dates)} dates using {strategy_name} after cleaning")
                        return parsed
                except:
                    continue
        except Exception as e:
            print(f"Skipping cleaning due to error: {str(e)}")
        
        # Last resort: return series with all NaT (skipping all problematic data)
        print(f"Warning: Could not parse any dates, skipping all {len(expanded_series)} rows")
        return pd.Series([pd.NaT] * len(expanded_series), dtype='datetime64[ns]')
    
    def _clean_date_strings(self, date_series):
        """Clean date strings to make them more parseable"""
        cleaned = date_series.astype(str)
        
        # Remove extra whitespace
        cleaned = cleaned.str.strip()
        
        # Replace common separators
        cleaned = cleaned.str.replace('\\', '/')
        cleaned = cleaned.str.replace('_', '-')
        
        # Handle common issues
        # If it looks like YYYYMMDD, add separators
        mask = cleaned.str.match(r'^\d{8}$')
        if mask.any():
            cleaned.loc[mask] = cleaned.loc[mask].str[:4] + '-' + cleaned.loc[mask].str[4:6] + '-' + cleaned.loc[mask].str[6:8]
        
        # If it looks like MMDDYYYY, rearrange
        mask = cleaned.str.match(r'^\d{2}\d{2}\d{4}$')
        if mask.any():
            cleaned.loc[mask] = cleaned.loc[mask].str[4:8] + '-' + cleaned.loc[mask].str[:2] + '-' + cleaned.loc[mask].str[2:4]
        
        # Handle ambiguous MM/DD or DD/MM formats by assuming current year
        mask = cleaned.str.match(r'^\d{1,2}/\d{1,2}$')
        if mask.any():
            current_year = pd.Timestamp.now().year
            cleaned.loc[mask] = cleaned.loc[mask] + f'/{current_year}'
        
        # Handle ambiguous MM-DD or DD-MM formats by assuming current year
        mask = cleaned.str.match(r'^\d{1,2}-\d{1,2}$')
        if mask.any():
            current_year = pd.Timestamp.now().year
            cleaned.loc[mask] = cleaned.loc[mask] + f'-{current_year}'
        
        return cleaned
    
    def _expand_ambiguous_dates(self, date_series):
        """Expand ambiguous date formats by adding current year"""
        expanded = date_series.astype(str)
        
        # Handle ambiguous MM/DD or DD/MM formats by assuming current year
        mask = expanded.str.match(r'^\d{1,2}/\d{1,2}$')
        if mask.any():
            current_year = pd.Timestamp.now().year
            expanded.loc[mask] = expanded.loc[mask] + f'/{current_year}'
            print(f"Expanded {mask.sum()} ambiguous MM/DD dates with year {current_year}")
        
        # Handle ambiguous MM-DD or DD-MM formats by assuming current year
        mask = expanded.str.match(r'^\d{1,2}-\d{1,2}$')
        if mask.any():
            current_year = pd.Timestamp.now().year
            expanded.loc[mask] = expanded.loc[mask] + f'-{current_year}'
            print(f"Expanded {mask.sum()} ambiguous MM-DD dates with year {current_year}")
        
        return expanded
    
    def validate_date_column(self, date_series):
        """Validate if a date column can be parsed"""
        try:
            parsed = self.parse_date_column(date_series)
            valid_count = parsed.notna().sum()
            total_count = len(date_series)
            
            if valid_count == 0:
                return False, "No valid dates found"
            elif valid_count < total_count * 0.8:  # Less than 80% valid
                return False, f"Only {valid_count}/{total_count} dates are valid"
            else:
                return True, f"Successfully parsed {valid_count}/{total_count} dates"
                
        except Exception as e:
            return False, f"Date parsing error: {str(e)}"
    
    def normalize_date_format(self, date_series, target_format='%Y-%m-%d'):
        """Parse dates and return them in a consistent format"""
        parsed = self.parse_date_column(date_series)
        return parsed.dt.strftime(target_format)
    
    def get_date_range(self, date_series):
        """Get the date range from a parsed date series"""
        parsed = self.parse_date_column(date_series)
        valid_dates = parsed.dropna()
        
        if len(valid_dates) == 0:
            return None, None
        
        return valid_dates.min(), valid_dates.max()
