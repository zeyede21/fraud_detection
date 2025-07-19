import pandas as pd
from utils import ip_to_int

def load_data(fraud_path, ip_path):
    # Load your fraud transaction data
    fraud_df = pd.read_csv(fraud_path)

    # Load the IP-to-country mapping data
    ip_df = pd.read_csv(ip_path)

    # Apply the conversion for IP addresses
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    ip_df['lower'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
    ip_df['upper'] = ip_df['upper_bound_ip_address'].apply(ip_to_int)

    return fraud_df, ip_df  # Return only fraud_df and ip_df

def clean_fraud_data(df):
    df = df.drop_duplicates()
    # Corrected columns
    df = df.dropna(subset=['user_id', 'purchase_value', 'class'])
    return df

def clean_ip_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['lower_bound_ip_address', 'country'])
    return df