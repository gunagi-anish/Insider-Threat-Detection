import pandas as pd

def label_users(user_df):
    # Assign label: 1 = malicious, 0 = benign
    user_df['label'] = user_df['user_role'].apply(lambda x: 1 if x == 'malicious' else 0)
    return user_df[['user', 'label']]
