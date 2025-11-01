import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr

try:
    from scipy.stats import ConstantInputWarning  # type: ignore
except ImportError:  # pragma: no cover
    ConstantInputWarning = RuntimeWarning

class LagDetector:
    def __init__(self, max_lag=10):
        self.max_lag = max_lag
        self.lag_results = {}
        
    def detect_lag(self, y_true, y_pred, model_name):
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        correlations = []
        lags = range(-self.max_lag, self.max_lag + 1)
        
        lags_list = list(lags)

        for lag in lags_list:
            if lag < 0:
                true_slice = y_true_array[:lag]
                pred_slice = y_pred_array[-lag:]
            elif lag > 0:
                true_slice = y_true_array[lag:]
                pred_slice = y_pred_array[:-lag]
            else:
                true_slice = y_true_array
                pred_slice = y_pred_array

            if len(true_slice) < 2 or len(pred_slice) < 2:
                correlations.append(0.0)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConstantInputWarning)
                try:
                    corr, _ = pearsonr(true_slice, pred_slice)
                except Exception:
                    corr = 0.0

            if not np.isfinite(corr):
                corr = 0.0

            correlations.append(float(corr))
        
        correlations = np.array(correlations)

        if correlations.size == 0:
            detected_lag = 0
            max_correlation = 0.0
        else:
            max_corr_value = float(np.max(correlations))
            candidate_indices = np.where(np.isclose(correlations, max_corr_value))[0]
            zero_lag_index = self.max_lag if self.max_lag < correlations.size else None

            if zero_lag_index is not None and zero_lag_index in candidate_indices:
                max_corr_idx = zero_lag_index
            else:
                max_corr_idx = int(candidate_indices[0])

            detected_lag = lags_list[max_corr_idx]
            max_correlation = float(correlations[max_corr_idx])
        
        zero_lag_correlation = float(correlations[self.max_lag]) if correlations.size else 0.0
        
        is_lagged = abs(detected_lag) > 0 and max_correlation > zero_lag_correlation + 0.05
        
        self.lag_results[model_name] = {
            'detected_lag': detected_lag,
            'max_correlation': max_correlation,
            'zero_lag_correlation': zero_lag_correlation,
            'is_lagged': is_lagged,
            'all_correlations': dict(zip(lags, correlations)),
            'correlation_improvement': max_correlation - zero_lag_correlation if is_lagged else 0
        }
        
        return self.lag_results[model_name]
    
    def get_lag_summary(self):
        if not self.lag_results:
            return None
        
        summary = []
        for model_name, results in self.lag_results.items():
            summary.append({
                'Model': model_name,
                'Detected Lag': results['detected_lag'],
                'Max Correlation': f"{results['max_correlation']:.4f}",
                'Zero-Lag Correlation': f"{results['zero_lag_correlation']:.4f}",
                'Is Lagged': 'Yes' if results['is_lagged'] else 'No',
                'Correlation Improvement': f"{results['correlation_improvement']:.4f}"
            })
        
        return pd.DataFrame(summary)
    
    def identify_false_positives(self, threshold=0.05):
        false_positives = []
        
        for model_name, results in self.lag_results.items():
            if results['is_lagged'] and results['correlation_improvement'] > threshold:
                false_positives.append({
                    'model': model_name,
                    'lag': results['detected_lag'],
                    'severity': 'High' if results['correlation_improvement'] > 0.15 else 'Medium',
                    'improvement': results['correlation_improvement']
                })
        
        return false_positives
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        if len(y_true_array) < 2:
            return 0
        
        true_direction = np.diff(y_true_array) > 0
        pred_direction = np.diff(y_pred_array) > 0
        
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        return directional_accuracy
    
    def analyze_prediction_shift(self, y_true, y_pred, model_name):
        shift_analysis = {}
        
        for shift in range(1, self.max_lag + 1):
            shifted_pred = np.roll(y_pred, shift)
            correlation, _ = pearsonr(y_true, shifted_pred)
            shift_analysis[shift] = correlation
        
        best_shift = max(shift_analysis.items(), key=lambda x: x[1])
        
        original_corr, _ = pearsonr(y_true, y_pred)
        
        return {
            'model': model_name,
            'best_shift': best_shift[0],
            'best_shift_correlation': best_shift[1],
            'original_correlation': original_corr,
            'is_shift_better': best_shift[1] > original_corr + 0.05
        }


def detect_pattern_memorization(y_true, y_pred, window=5):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    memorization_scores = []
    
    for i in range(len(y_true_array) - window):
        true_window = y_true_array[i:i+window]
        pred_window = y_pred_array[i:i+window]
        
        true_pattern = np.diff(true_window) > 0
        pred_pattern = np.diff(pred_window) > 0
        
        pattern_match = np.mean(true_pattern == pred_pattern)
        memorization_scores.append(pattern_match)
    
    avg_memorization = np.mean(memorization_scores) if memorization_scores else 0
    
    return {
        'avg_pattern_match': avg_memorization,
        'is_memorizing': avg_memorization > 0.9,
        'pattern_consistency': np.std(memorization_scores) if memorization_scores else 0
    }


def calculate_prediction_variance(predictions_dict):
    predictions_array = np.array([pred for pred in predictions_dict.values()])
    
    variance_per_sample = np.var(predictions_array, axis=0)
    
    return {
        'mean_variance': np.mean(variance_per_sample),
        'max_variance': np.max(variance_per_sample),
        'variance_distribution': variance_per_sample
    }
