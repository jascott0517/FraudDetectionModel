import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_model(X_train, y_train):
    """Train XGBoost model"""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, features):
    """Evaluate model performance"""
    y_pred = model.predict_proba(X_test)[:,1]
    
    # Metrics
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred > 0.5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred > 0.5)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title('Feature Importance')
    plt.show()
    
    return y_pred

def save_artifacts(model, scaler, features, path='../artifacts'):
    """Save model artifacts"""
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f'{path}/model.pkl')
    joblib.dump(scaler, f'{path}/scaler.pkl')
    joblib.dump(features, f'{path}/features.pkl')

def load_artifacts(path='../artifacts'):
    """Load model artifacts"""
    model = joblib.load(f'{path}/model.pkl')
    scaler = joblib.load(f'{path}/scaler.pkl')
    features = joblib.load(f'{path}/features.pkl')
    return model, scaler, features