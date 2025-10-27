import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RandomWalkModel:
    def __init__(self):
        self.name = "Random Walk"
        self.predictions = None
        self.last_train_value = None
        self.last_train_direction = None
        
    def fit(self, X_train, y_train, prices_train=None):
        if prices_train is not None and len(prices_train) > 0:
            self.last_train_value = prices_train.iloc[-1]
        elif hasattr(y_train, 'iloc') and len(y_train) > 0:
            self.last_train_value = y_train.iloc[-1]
        else:
            self.last_train_value = 0
        
        if hasattr(y_train, 'iloc') and len(y_train) > 0:
            self.last_train_direction = int(y_train.iloc[-1])
        else:
            self.last_train_direction = 1
    
    def predict(self, X_test, last_train_price=None):
        if last_train_price is not None:
            predictions = np.full(len(X_test), last_train_price)
        elif self.last_train_value is not None:
            predictions = np.full(len(X_test), self.last_train_value)
        else:
            predictions = np.zeros(len(X_test))
            
        return predictions
    
    def predict_classification(self, X_test):
        if self.last_train_direction is not None:
            predictions = np.full(len(X_test), self.last_train_direction)
        else:
            predictions = np.ones(len(X_test))
        return predictions.astype(int)
    
    def get_params(self):
        return {
            'model_type': 'Baseline',
            'description': 'Predicts that tomorrow\'s price will equal today\'s price (no change)',
            'parameters': {
                'last_value': self.last_train_value,
                'last_direction': self.last_train_direction
            }
        }


class SimpleLinearRegressionModel:
    def __init__(self):
        self.name = "Linear Regression (Baseline)"
        self.model_regression = LinearRegression()
        self.model_classification = LinearRegression()
        self.fitted = False
        
    def fit(self, X_train, y_train_class, y_train_reg):
        self.model_regression.fit(X_train, y_train_reg)
        self.model_classification.fit(X_train, y_train_class)
        self.fitted = True
        
    def predict_regression(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model_regression.predict(X_test)
    
    def predict_classification(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        preds = self.model_classification.predict(X_test)
        return (preds > 0.5).astype(int)
    
    def get_feature_importance(self, feature_names):
        if not self.fitted:
            return None
            
        importance = np.abs(self.model_regression.coef_)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_params(self):
        return {
            'model_type': 'Baseline',
            'description': 'Simple linear regression using all features',
            'parameters': {
                'fit_intercept': True,
                'n_features': len(self.model_regression.coef_) if self.fitted else 0
            }
        }
    
    def get_metrics(self, y_true, y_pred):
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
