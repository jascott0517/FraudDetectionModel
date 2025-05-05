import pandas as pd
import numpy as np
from textstat import flesch_reading_ease
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
import html5lib
from tqdm import tqdm
tqdm.pandas()

def load_and_merge_data(data_path='../data/'):
    """Load and merge fraud and fund datasets"""
    fraud_train = pd.read_csv(f'{data_path}/fraud_data_train.csv')
    fraud_test = pd.read_csv(f'{data_path}/fraud_data_test.csv')
    fund_train = pd.read_csv(f'{data_path}/fund_data_train.csv')
    fund_test = pd.read_csv(f'{data_path}/fund_data_test.csv')
    
    train = pd.merge(fraud_train, fund_train, on='fund_id')
    test = pd.merge(fraud_test, fund_test, on='fund_id')
    
    return train, test

def clean_html(text):
    """Remove HTML tags from text"""
    if pd.isna(text):
        return ""
    return str(text).replace('<[^<]+?>', '')

def extract_text_features(df):
    """Extract features from text columns"""
    df['clean_description'] = df['description'].progress_apply(clean_html)
    df['desc_readability'] = df['clean_description'].progress_apply(flesch_reading_ease)
    df['desc_sentiment'] = df['clean_description'].progress_apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['desc_word_count'] = df['clean_description'].progress_apply(lambda x: len(str(x).split()))
    return df

def preprocess_data(train, test):
    """Preprocess the data for modeling"""
    # Process text features
    train = extract_text_features(train)
    test = extract_text_features(test)
    
    # Handle categorical features
    train['is_disposable'] = train['primary_email_address_checks__is_disposable'].astype(int)
    test['is_disposable'] = test['primary_email_address_checks__is_disposable'].astype(int)
    
    # Select features
    features = [
        'goal', 'descr_len', 'title_len', 'identity_check_score',
        'is_disposable', 'primary_email_address_checks__email_domain_creation_days',
        'desc_readability', 'desc_sentiment', 'desc_word_count'
    ]
    
    X_train = train[features]
    y_train = train['label']
    X_test = test[features]
    y_test = test['label']
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, features