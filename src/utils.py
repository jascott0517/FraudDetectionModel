import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def add_text_features(X_train, X_test, train_text, test_text):
    """Add TF-IDF text features to existing features"""
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_train_text = vectorizer.fit_transform(train_text)
    X_test_text = vectorizer.transform(test_text)
    
    X_train_full = hstack([X_train, X_train_text])
    X_test_full = hstack([X_test, X_test_text])
    
    return X_train_full, X_test_full, vectorizer

def generate_report(y_true, y_pred, threshold=0.5):
    """Generate performance report"""
    from sklearn.metrics import precision_recall_fscore_support
    report = {}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred > threshold, average='binary')
    
    report['precision'] = precision
    report['recall'] = recall
    report['f1'] = f1
    report['threshold'] = threshold
    
    return pd.DataFrame([report])