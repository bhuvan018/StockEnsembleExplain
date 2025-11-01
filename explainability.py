import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

class ExplainabilityAnalyzer:
    def __init__(self):
        self.feature_importance_cache = {}
        self.shap_style_values = {}
        
    def get_feature_importance(self, model, X, y, feature_names, method='default'):
        model_name = model.name if hasattr(model, 'name') else str(type(model).__name__)
        
        if method == 'permutation':
            return self._permutation_importance(model, X, y, feature_names)
        
        if hasattr(model, 'get_feature_importance'):
            importance_df = None
            getter = getattr(model, 'get_feature_importance')

            # Try to call with feature names first (for models that expect them),
            # then fallback to the original signature if provided.
            try:
                if feature_names is not None:
                    importance_df = getter(feature_names)
                else:
                    importance_df = getter()
            except TypeError:
                importance_df = getter()

            if importance_df is not None:
                return importance_df
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return importance_df
        
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]
            importance = np.abs(coef)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return importance_df
        
        return self._permutation_importance(model, X, y, feature_names)
    
    def _permutation_importance(self, model, X, y, feature_names, n_repeats=10):
        try:
            if hasattr(model, 'model'):
                perm_importance = permutation_importance(
                    model.model, X, y, 
                    n_repeats=n_repeats, 
                    random_state=42,
                    n_jobs=-1
                )
            else:
                perm_importance = permutation_importance(
                    model, X, y, 
                    n_repeats=n_repeats, 
                    random_state=42,
                    n_jobs=-1
                )
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            return importance_df
        except Exception as e:
            print(f"Error calculating permutation importance: {e}")
            return pd.DataFrame({
                'feature': feature_names,
                'importance': np.zeros(len(feature_names))
            })
    
    def calculate_prediction_confidence(self, model, X, predictions):
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                if proba is not None:
                    confidence = np.max(proba, axis=1)
                    return confidence
            except:
                pass
        
        return np.ones(len(predictions)) * 0.5
    
    def analyze_feature_correlations(self, X, feature_names, top_n=10):
        corr_matrix = X.corr()
        
        highly_correlated = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    highly_correlated.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return pd.DataFrame(highly_correlated).sort_values('correlation', ascending=False)
    
    def calculate_feature_impact(self, model, X, y, feature_names, sample_size=100):
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
        
        feature_impacts = {}
        
        for feature in feature_names:
            X_modified = X_sample.copy()
            original_predictions = model.predict(X_sample)
            
            X_modified[feature] = X_sample[feature].mean()
            modified_predictions = model.predict(X_modified)
            
            impact = np.mean(np.abs(original_predictions - modified_predictions))
            feature_impacts[feature] = impact
        
        impact_df = pd.DataFrame({
            'feature': list(feature_impacts.keys()),
            'impact': list(feature_impacts.values())
        }).sort_values('impact', ascending=False)
        
        return impact_df
    
    def get_top_features_by_model(self, models_dict, X, y, feature_names, top_n=10):
        all_importances = {}
        
        for model_name, model in models_dict.items():
            importance_df = self.get_feature_importance(model, X, y, feature_names)
            if importance_df is not None and not importance_df.empty:
                all_importances[model_name] = importance_df.head(top_n)
        
        return all_importances
    
    def create_feature_importance_summary(self, models_dict, X, y, feature_names):
        summary = pd.DataFrame({'feature': feature_names})
        
        for model_name, model in models_dict.items():
            importance_df = self.get_feature_importance(model, X, y, feature_names)
            
            if importance_df is not None and not importance_df.empty:
                importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
                summary[model_name] = summary['feature'].map(importance_dict).fillna(0)
        
        summary['mean_importance'] = summary.iloc[:, 1:].mean(axis=1)
        summary['std_importance'] = summary.iloc[:, 1:-1].std(axis=1)
        summary = summary.sort_values('mean_importance', ascending=False)
        
        return summary
    
    def analyze_prediction_contributions(self, model, X, prediction_idx, feature_names, baseline_prediction=None):
        if len(X) <= prediction_idx:
            return None
        
        sample = X.iloc[prediction_idx:prediction_idx+1]
        
        if baseline_prediction is None:
            baseline_prediction = model.predict(pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns))[0]
        
        actual_prediction = model.predict(sample)[0]
        
        contributions = {}
        for feature in feature_names:
            X_modified = sample.copy()
            X_modified[feature] = 0
            modified_prediction = model.predict(X_modified)[0]
            
            contribution = actual_prediction - modified_prediction
            contributions[feature] = contribution
        
        contrib_df = pd.DataFrame({
            'feature': list(contributions.keys()),
            'contribution': list(contributions.values())
        }).sort_values('contribution', key=abs, ascending=False)
        
        return {
            'baseline_prediction': baseline_prediction,
            'actual_prediction': actual_prediction,
            'contributions': contrib_df
        }


def calculate_partial_dependence(model, X, feature, num_points=50):
    feature_values = np.linspace(X[feature].min(), X[feature].max(), num_points)
    
    partial_predictions = []
    
    for value in feature_values:
        X_modified = X.copy()
        X_modified[feature] = value
        
        predictions = model.predict(X_modified)
        avg_prediction = np.mean(predictions)
        partial_predictions.append(avg_prediction)
    
    return feature_values, np.array(partial_predictions)


def identify_key_decision_factors(model, X, y, feature_names, threshold=0.1):
    explainer = ExplainabilityAnalyzer()
    importance_df = explainer.get_feature_importance(model, X, y, feature_names)
    
    if importance_df is None or importance_df.empty:
        return []
    
    max_importance = importance_df['importance'].max()
    key_features = importance_df[importance_df['importance'] >= max_importance * threshold]
    
    return key_features['feature'].tolist()


def calculate_shap_style_values(model, X, feature_names, sample_size=100):
    if len(X) > sample_size:
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        X_sample = X
    
    baseline_mean = X.mean()
    X_baseline = pd.DataFrame([baseline_mean] * len(X_sample), columns=X.columns, index=X_sample.index)
    
    baseline_predictions = model.predict(X_baseline)
    actual_predictions = model.predict(X_sample)
    
    shap_values_matrix = np.zeros((len(X_sample), len(feature_names)))
    
    for i, feature in enumerate(feature_names):
        X_modified = X_sample.copy()
        X_modified[feature] = baseline_mean[feature]
        
        modified_predictions = model.predict(X_modified)
        shap_values_matrix[:, i] = actual_predictions - modified_predictions
    
    shap_values_df = pd.DataFrame(
        shap_values_matrix,
        columns=feature_names,
        index=X_sample.index
    )
    
    return {
        'shap_values': shap_values_df,
        'baseline_predictions': baseline_predictions,
        'actual_predictions': actual_predictions,
        'feature_values': X_sample
    }


def get_shap_style_summary(shap_results, top_n=15):
    shap_values = shap_results['shap_values']
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    summary_df = pd.DataFrame({
        'feature': shap_values.columns,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    return summary_df.head(top_n)


def get_waterfall_data(shap_results, sample_idx=0):
    shap_values = shap_results['shap_values'].iloc[sample_idx]
    feature_values = shap_results['feature_values'].iloc[sample_idx]
    baseline = shap_results['baseline_predictions'][sample_idx]
    prediction = shap_results['actual_predictions'][sample_idx]
    
    contributions = pd.DataFrame({
        'feature': shap_values.index,
        'contribution': shap_values.values,
        'feature_value': feature_values.values
    }).sort_values('contribution', key=abs, ascending=False)
    
    top_contributions = contributions.head(10)
    
    return {
        'baseline': baseline,
        'prediction': prediction,
        'contributions': top_contributions
    }
