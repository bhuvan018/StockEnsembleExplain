import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

class WeightedVotingEnsemble:
    def __init__(self, task='classification'):
        self.name = "Weighted Voting Ensemble"
        self.task = task
        self.weights = {}
        self.models = []
        
    def set_weights(self, model_names, weights):
        if len(model_names) != len(weights):
            raise ValueError("Number of models and weights must match")
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        self.weights = dict(zip(model_names, normalized_weights))
    
    def predict(self, predictions_dict):
        if not self.weights:
            equal_weight = 1.0 / len(predictions_dict)
            self.weights = {name: equal_weight for name in predictions_dict.keys()}
        
        if self.task == 'classification':
            weighted_sum = np.zeros(len(next(iter(predictions_dict.values()))))
            
            for model_name, predictions in predictions_dict.items():
                weight = self.weights.get(model_name, 0)
                weighted_sum += predictions * weight
            
            ensemble_predictions = (weighted_sum > 0.5).astype(int)
        else:
            weighted_sum = np.zeros(len(next(iter(predictions_dict.values()))))
            
            for model_name, predictions in predictions_dict.items():
                weight = self.weights.get(model_name, 0)
                weighted_sum += predictions * weight
            
            ensemble_predictions = weighted_sum
        
        return ensemble_predictions
    
    def get_params(self):
        return {
            'model_type': 'Ensemble',
            'method': 'Weighted Voting',
            'weights': self.weights
        }


class StackingEnsemble:
    def __init__(self, task='classification'):
        self.name = "Stacking Ensemble"
        self.task = task
        
        if task == 'classification':
            self.meta_model = LogisticRegression(max_iter=1000)
        else:
            self.meta_model = LinearRegression()
        
        self.fitted = False
        
    def fit(self, predictions_dict_train, y_train):
        stacked_features = np.column_stack([pred for pred in predictions_dict_train.values()])
        
        self.meta_model.fit(stacked_features, y_train)
        self.fitted = True
        self.model_names = list(predictions_dict_train.keys())
    
    def predict(self, predictions_dict_test):
        if not self.fitted:
            raise ValueError("Meta-model must be fitted before prediction")
        
        stacked_features = np.column_stack([predictions_dict_test[name] for name in self.model_names])
        
        if self.task == 'classification':
            return self.meta_model.predict(stacked_features)
        else:
            return self.meta_model.predict(stacked_features)
    
    def get_meta_model_weights(self):
        if not self.fitted:
            return None
        
        if hasattr(self.meta_model, 'coef_'):
            weights = self.meta_model.coef_
            if len(weights.shape) > 1:
                weights = weights[0]
            
            weight_dict = dict(zip(self.model_names, weights))
            return weight_dict
        
        return None
    
    def get_params(self):
        return {
            'model_type': 'Ensemble',
            'method': 'Stacking',
            'meta_model': type(self.meta_model).__name__,
            'weights': self.get_meta_model_weights()
        }


class DynamicWeightingEnsemble:
    def __init__(self, task='classification', window_size=20):
        self.name = "Dynamic Weighting Ensemble"
        self.task = task
        self.window_size = window_size
        self.performance_history = {}
        self.current_weights = {}
        
    def update_performance(self, model_name, y_true, y_pred):
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        if self.task == 'classification':
            score = accuracy_score(y_true, y_pred)
        else:
            score = 1.0 / (1.0 + mean_squared_error(y_true, y_pred))
        
        self.performance_history[model_name].append(score)
        
        if len(self.performance_history[model_name]) > self.window_size:
            self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
    
    def calculate_weights(self):
        if not self.performance_history:
            return {}
        
        avg_performance = {}
        for model_name, scores in self.performance_history.items():
            avg_performance[model_name] = np.mean(scores) if scores else 0
        
        total_performance = sum(avg_performance.values())
        
        if total_performance > 0:
            self.current_weights = {
                name: perf / total_performance 
                for name, perf in avg_performance.items()
            }
        else:
            n_models = len(avg_performance)
            self.current_weights = {name: 1.0 / n_models for name in avg_performance.keys()}
        
        return self.current_weights
    
    def predict(self, predictions_dict):
        if not self.current_weights:
            self.current_weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict.keys()}
        
        if self.task == 'classification':
            weighted_sum = np.zeros(len(next(iter(predictions_dict.values()))))
            
            for model_name, predictions in predictions_dict.items():
                weight = self.current_weights.get(model_name, 0)
                weighted_sum += predictions * weight
            
            ensemble_predictions = (weighted_sum > 0.5).astype(int)
        else:
            weighted_sum = np.zeros(len(next(iter(predictions_dict.values()))))
            
            for model_name, predictions in predictions_dict.items():
                weight = self.current_weights.get(model_name, 0)
                weighted_sum += predictions * weight
            
            ensemble_predictions = weighted_sum
        
        return ensemble_predictions
    
    def get_performance_summary(self):
        summary = {}
        for model_name, scores in self.performance_history.items():
            summary[model_name] = {
                'avg_score': np.mean(scores) if scores else 0,
                'recent_score': scores[-1] if scores else 0,
                'current_weight': self.current_weights.get(model_name, 0),
                'n_evaluations': len(scores)
            }
        return summary
    
    def get_params(self):
        return {
            'model_type': 'Ensemble',
            'method': 'Dynamic Weighting',
            'window_size': self.window_size,
            'current_weights': self.current_weights,
            'performance_summary': self.get_performance_summary()
        }


def calculate_model_agreement(predictions_dict):
    predictions_array = np.array([pred for pred in predictions_dict.values()])
    
    agreement_scores = []
    n_samples = predictions_array.shape[1]
    
    for i in range(n_samples):
        sample_predictions = predictions_array[:, i]
        unique_predictions = np.unique(sample_predictions)
        
        if len(unique_predictions) == 1:
            agreement_scores.append(1.0)
        else:
            most_common = np.bincount(sample_predictions.astype(int)).max()
            agreement_scores.append(most_common / len(sample_predictions))
    
    return np.array(agreement_scores)


def calculate_ensemble_confidence(predictions_dict, task='classification'):
    if task == 'classification':
        predictions_array = np.array([pred for pred in predictions_dict.values()])
        
        confidence_scores = []
        n_samples = predictions_array.shape[1]
        
        for i in range(n_samples):
            sample_predictions = predictions_array[:, i]
            positive_ratio = np.mean(sample_predictions)
            confidence = max(positive_ratio, 1 - positive_ratio)
            confidence_scores.append(confidence)
        
        return np.array(confidence_scores)
    else:
        predictions_array = np.array([pred for pred in predictions_dict.values()])
        std_scores = np.std(predictions_array, axis=0)
        max_std = np.max(std_scores) if len(std_scores) > 0 else 1
        
        if max_std > 0:
            confidence_scores = 1 - (std_scores / max_std)
        else:
            confidence_scores = np.ones(predictions_array.shape[1])
        
        return confidence_scores
