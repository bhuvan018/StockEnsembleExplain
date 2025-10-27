import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

class LogisticRegressionModel:
    def __init__(self, max_iter=1000, random_state=42):
        self.name = "Logistic Regression"
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.fitted = False
        self.feature_names = None
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.fitted = True
        
    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self):
        if not self.fitted or self.feature_names is None:
            return None
            
        importance = np.abs(self.model.coef_[0])
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_permutation_importance(self, X, y):
        if not self.fitted:
            return None
            
        perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_params(self):
        return {
            'model_type': 'Traditional ML',
            'algorithm': 'Logistic Regression',
            'parameters': self.model.get_params()
        }


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42, task='classification'):
        self.name = f"Random Forest ({'Classification' if task == 'classification' else 'Regression'})"
        self.task = task
        
        if task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        
        self.fitted = False
        self.feature_names = None
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.fitted = True
        
    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if not self.fitted or self.task != 'classification':
            return None
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self):
        if not self.fitted or self.feature_names is None:
            return None
            
        importance = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_params(self):
        return {
            'model_type': 'Traditional ML',
            'algorithm': 'Random Forest',
            'task': self.task,
            'parameters': self.model.get_params()
        }


class SVMModel:
    def __init__(self, kernel='rbf', C=1.0, random_state=42, task='classification'):
        self.name = f"SVM ({'Classification' if task == 'classification' else 'Regression'})"
        self.task = task
        
        if task == 'classification':
            self.model = SVC(
                kernel=kernel,
                C=C,
                random_state=random_state,
                probability=True
            )
        else:
            self.model = SVR(
                kernel=kernel,
                C=C
            )
        
        self.fitted = False
        self.feature_names = None
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.fitted = True
        
    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if not self.fitted or self.task != 'classification':
            return None
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self, X, y):
        if not self.fitted or self.feature_names is None:
            return None
            
        perm_importance = permutation_importance(self.model, X, y, n_repeats=10, random_state=42)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_params(self):
        return {
            'model_type': 'Traditional ML',
            'algorithm': 'SVM',
            'task': self.task,
            'parameters': self.model.get_params()
        }


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
        from sklearn.metrics import roc_auc_score
        try:
            metrics['ROC AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            pass
    
    return metrics


def calculate_regression_metrics(y_true, y_pred):
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        metrics['MAPE'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    
    return metrics


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    if len(returns) == 0:
        return 0
    
    excess_returns = returns - risk_free_rate / 252
    
    if np.std(excess_returns) == 0:
        return 0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe
