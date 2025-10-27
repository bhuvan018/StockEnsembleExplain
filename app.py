import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_utils import StockDataFetcher
from baseline_models import RandomWalkModel, SimpleLinearRegressionModel
from traditional_ml_models import (
    LogisticRegressionModel, RandomForestModel, SVMModel,
    calculate_classification_metrics, calculate_regression_metrics, calculate_sharpe_ratio
)
from deep_learning_models import CNNModel, GRUModel, LSTMModel
from ensemble_methods import (
    WeightedVotingEnsemble, StackingEnsemble, DynamicWeightingEnsemble,
    calculate_model_agreement, calculate_ensemble_confidence
)
from lag_detection import LagDetector, detect_pattern_memorization, calculate_prediction_variance
from explainability import (
    ExplainabilityAnalyzer, calculate_partial_dependence, identify_key_decision_factors,
    calculate_shap_style_values, get_shap_style_summary, get_waterfall_data
)

st.set_page_config(
    page_title="Ensemble Stock Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_stock_data(ticker, lookback_days):
    try:
        fetcher = StockDataFetcher(ticker, lookback_days=lookback_days)
        fetcher.fetch_data()
        fetcher.calculate_technical_indicators()
        data_split = fetcher.get_train_test_split(test_size=0.2)
        summary = fetcher.get_data_summary()
        return data_split, summary, fetcher
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def train_all_models(data_split, task='classification'):
    models = {}
    predictions_train = {}
    predictions_test = {}
    
    X_train = data_split['X_train_scaled']
    X_test = data_split['X_test_scaled']
    y_train = data_split['y_train_class'] if task == 'classification' else data_split['y_train_reg']
    y_test = data_split['y_test_class'] if task == 'classification' else data_split['y_test_reg']
    
    with st.spinner("Training baseline models..."):
        rw_model = RandomWalkModel()
        rw_model.fit(X_train, y_train, data_split['prices_train'])
        models['Random Walk'] = rw_model
        
        lr_model = SimpleLinearRegressionModel()
        lr_model.fit(X_train, data_split['y_train_class'], data_split['y_train_reg'])
        models['Linear Regression'] = lr_model
    
    with st.spinner("Training traditional ML models..."):
        log_reg = LogisticRegressionModel()
        log_reg.fit(X_train, y_train)
        models['Logistic Regression'] = log_reg
        
        rf_model = RandomForestModel(n_estimators=50, max_depth=10, task=task)
        rf_model.fit(X_train, y_train)
        models['Random Forest'] = rf_model
        
        svm_model = SVMModel(task=task)
        svm_model.fit(X_train, y_train)
        models['SVM'] = svm_model
    
    with st.spinner("Training deep learning models..."):
        lookback = 10
        n_features = X_train.shape[1] // lookback
        
        if X_train.shape[1] % lookback != 0:
            pad_size = lookback - (X_train.shape[1] % lookback)
            X_train_padded = np.pad(X_train.values, ((0, 0), (0, pad_size)), mode='constant')
            X_test_padded = np.pad(X_test.values, ((0, 0), (0, pad_size)), mode='constant')
            X_train = pd.DataFrame(X_train_padded, index=X_train.index)
            X_test = pd.DataFrame(X_test_padded, index=X_test.index)
        
        gru_model = GRUModel(input_shape=X_train.shape[1], task=task, lookback=lookback)
        gru_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        models['GRU'] = gru_model
        
        lstm_model = LSTMModel(input_shape=X_train.shape[1], task=task, lookback=lookback)
        lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        models['LSTM'] = lstm_model
    
    for name, model in models.items():
        if name == 'Random Walk':
            if task == 'classification':
                predictions_train[name] = model.predict_classification(X_train)
                predictions_test[name] = model.predict_classification(X_test)
            else:
                last_train_price = data_split['prices_train'].iloc[-1]
                predictions_train[name] = model.predict(X_train, last_train_price)
                predictions_test[name] = model.predict(X_test, last_train_price)
        elif name == 'Linear Regression':
            if task == 'classification':
                predictions_train[name] = model.predict_classification(X_train)
                predictions_test[name] = model.predict_classification(X_test)
            else:
                predictions_train[name] = model.predict_regression(X_train)
                predictions_test[name] = model.predict_regression(X_test)
        else:
            predictions_train[name] = model.predict(X_train)
            predictions_test[name] = model.predict(X_test)
    
    return models, predictions_train, predictions_test

def calculate_all_metrics(predictions_dict, y_true, task='classification'):
    metrics_df = []
    
    for model_name, predictions in predictions_dict.items():
        if task == 'classification':
            metrics = calculate_classification_metrics(y_true, predictions)
        else:
            metrics = calculate_regression_metrics(y_true, predictions)
        
        metrics['Model'] = model_name
        metrics_df.append(metrics)
    
    return pd.DataFrame(metrics_df)

def main():
    st.title("📈 Ensemble Stock Prediction System")
    st.markdown("### Multi-Model Comparison with Explainability Analysis")
    
    st.sidebar.header("Configuration")
    
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)")
    lookback_days = st.sidebar.slider("Lookback Days", min_value=180, max_value=730, value=365, step=30)
    task = st.sidebar.selectbox("Prediction Task", ["Classification (Up/Down)", "Regression (Price)"], index=0)
    task_type = 'classification' if 'Classification' in task else 'regression'
    
    if st.sidebar.button("Load Data & Train Models", type="primary"):
        with st.spinner(f"Loading {ticker} data..."):
            data_split, summary, fetcher = load_stock_data(ticker, lookback_days)
            
            if data_split is None:
                st.error("Failed to load data. Please check the ticker symbol and try again.")
                return
            
            st.session_state['data_split'] = data_split
            st.session_state['summary'] = summary
            st.session_state['fetcher'] = fetcher
            st.session_state['ticker'] = ticker
        
        models, predictions_train, predictions_test = train_all_models(data_split, task=task_type)
        
        st.session_state['models'] = models
        st.session_state['predictions_train'] = predictions_train
        st.session_state['predictions_test'] = predictions_test
        st.session_state['task_type'] = task_type
        
        st.success("✅ Models trained successfully!")
    
    if 'data_split' not in st.session_state:
        st.info("👈 Configure settings in the sidebar and click 'Load Data & Train Models' to begin")
        
        st.markdown("---")
        st.markdown("### 🎯 System Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Multiple Model Types**")
            st.markdown("- Baseline: Random Walk, Linear Regression")
            st.markdown("- Traditional ML: Logistic Regression, Random Forest, SVM")
            st.markdown("- Deep Learning: CNN, GRU, LSTM")
        
        with col2:
            st.markdown("**🔍 Explainability Analysis**")
            st.markdown("- Feature importance rankings")
            st.markdown("- Permutation importance")
            st.markdown("- Prediction confidence intervals")
        
        with col3:
            st.markdown("**🎲 Ensemble Methods**")
            st.markdown("- Weighted voting")
            st.markdown("- Stacking with meta-learner")
            st.markdown("- Dynamic weighting based on performance")
        
        return
    
    data_split = st.session_state['data_split']
    summary = st.session_state['summary']
    models = st.session_state['models']
    predictions_train = st.session_state['predictions_train']
    predictions_test = st.session_state['predictions_test']
    task_type = st.session_state['task_type']
    ticker = st.session_state['ticker']
    
    tabs = st.tabs([
        "📊 Overview", 
        "📈 Model Comparison", 
        "🎲 Ensemble Analysis", 
        "🔍 Explainability",
        "⚠️ Lag Detection",
        "📉 Predictions"
    ])
    
    with tabs[0]:
        st.header(f"Data Overview: {ticker}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", summary['total_records'])
        with col2:
            st.metric("Features", summary['features_count'])
        with col3:
            st.metric("Price Range", summary['price_range'])
        with col4:
            st.metric("Avg Volume", f"{float(summary['avg_volume']):,.0f}")
        
        st.markdown("---")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_split['dates_train'],
            y=data_split['prices_train'],
            mode='lines',
            name='Training Data',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=data_split['dates_test'],
            y=data_split['prices_test'],
            mode='lines',
            name='Test Data',
            line=dict(color='orange', width=2)
        ))
        fig.update_layout(
            title=f"{ticker} Stock Price - Train/Test Split",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Set")
            st.write(f"**Size:** {len(data_split['X_train'])} samples")
            st.write(f"**Date Range:** {data_split['dates_train'][0].strftime('%Y-%m-%d')} to {data_split['dates_train'][-1].strftime('%Y-%m-%d')}")
        
        with col2:
            st.subheader("Test Set")
            st.write(f"**Size:** {len(data_split['X_test'])} samples")
            st.write(f"**Date Range:** {data_split['dates_test'][0].strftime('%Y-%m-%d')} to {data_split['dates_test'][-1].strftime('%Y-%m-%d')}")
    
    with tabs[1]:
        st.header("Model Performance Comparison")
        
        y_test = data_split['y_test_class'] if task_type == 'classification' else data_split['y_test_reg']
        metrics_df = calculate_all_metrics(predictions_test, y_test, task=task_type)
        
        st.subheader("Performance Metrics")
        
        metrics_display = metrics_df.set_index('Model')
        st.dataframe(
            metrics_display.style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("Metrics Visualization")
        
        metric_cols = [col for col in metrics_df.columns if col != 'Model']
        selected_metric = st.selectbox("Select Metric", metric_cols)
        
        fig = px.bar(
            metrics_df,
            x='Model',
            y=selected_metric,
            title=f"{selected_metric} by Model",
            color=selected_metric,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.header("Ensemble Analysis")
        
        y_test = data_split['y_test_class'] if task_type == 'classification' else data_split['y_test_reg']
        
        st.subheader("Ensemble Methods Comparison")
        
        weighted_ensemble = WeightedVotingEnsemble(task=task_type)
        weighted_pred = weighted_ensemble.predict(predictions_test)
        
        stacking_ensemble = StackingEnsemble(task=task_type)
        y_train = data_split['y_train_class'] if task_type == 'classification' else data_split['y_train_reg']
        stacking_ensemble.fit(predictions_train, y_train)
        stacking_pred = stacking_ensemble.predict(predictions_test)
        
        dynamic_ensemble = DynamicWeightingEnsemble(task=task_type, window_size=20)
        
        for i in range(0, len(y_test), 10):
            end_idx = min(i + 10, len(y_test))
            for model_name, preds in predictions_test.items():
                dynamic_ensemble.update_performance(model_name, y_test.iloc[i:end_idx], preds[i:end_idx])
        
        dynamic_ensemble.calculate_weights()
        dynamic_pred = dynamic_ensemble.predict(predictions_test)
        
        ensemble_predictions = {
            'Weighted Voting': weighted_pred,
            'Stacking': stacking_pred,
            'Dynamic Weighting': dynamic_pred
        }
        
        ensemble_metrics = calculate_all_metrics(ensemble_predictions, y_test, task=task_type)
        individual_metrics = calculate_all_metrics(predictions_test, y_test, task=task_type)
        
        all_metrics = pd.concat([individual_metrics, ensemble_metrics])
        all_metrics['Type'] = ['Individual'] * len(individual_metrics) + ['Ensemble'] * len(ensemble_metrics)
        
        st.dataframe(
            all_metrics.set_index('Model').style.format("{:.4f}").background_gradient(cmap='RdYlGn', axis=0),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("Ensemble Weights & Contribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Weighted Voting (Equal Weights)**")
            weights_df = pd.DataFrame({
                'Model': list(weighted_ensemble.weights.keys()),
                'Weight': list(weighted_ensemble.weights.values())
            })
            fig = px.pie(weights_df, values='Weight', names='Model', title='Model Weights')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Dynamic Weighting (Performance-Based)**")
            dynamic_weights = dynamic_ensemble.current_weights
            weights_df = pd.DataFrame({
                'Model': list(dynamic_weights.keys()),
                'Weight': list(dynamic_weights.values())
            })
            fig = px.pie(weights_df, values='Weight', names='Model', title='Dynamic Weights')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Model Agreement Analysis")
        
        agreement_scores = calculate_model_agreement(predictions_test)
        confidence_scores = calculate_ensemble_confidence(predictions_test, task=task_type)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(agreement_scores))),
            y=agreement_scores,
            mode='lines',
            name='Model Agreement',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(confidence_scores))),
            y=confidence_scores,
            mode='lines',
            name='Ensemble Confidence',
            line=dict(color='green')
        ))
        fig.update_layout(
            title="Model Agreement and Ensemble Confidence",
            xaxis_title="Sample Index",
            yaxis_title="Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.header("🔍 Explainability Analysis with SHAP-Style Visualizations")
        
        explainer = ExplainabilityAnalyzer()
        
        selected_model_name = st.selectbox("Select Model for Analysis", list(models.keys()))
        selected_model = models[selected_model_name]
        
        X_test = data_split['X_test_scaled']
        y_test = data_split['y_test_class'] if task_type == 'classification' else data_split['y_test_reg']
        feature_names = data_split['feature_names']
        
        analysis_tabs = st.tabs(["Feature Importance", "SHAP-Style Values", "Individual Predictions", "Partial Dependence"])
        
        with analysis_tabs[0]:
            st.subheader(f"📊 Feature Importance: {selected_model_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model-Specific Importance**")
                importance_df = explainer.get_feature_importance(
                    selected_model,
                    X_test,
                    y_test,
                    feature_names,
                    method='default'
                )
                
                if importance_df is not None and not importance_df.empty:
                    top_features = importance_df.head(15)
                    
                    fig = px.bar(
                        top_features,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f"Top 15 Features",
                        color='importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Feature importance not available")
            
            with col2:
                st.write("**Permutation Importance**")
                perm_importance_df = explainer.get_feature_importance(
                    selected_model,
                    X_test,
                    y_test,
                    feature_names,
                    method='permutation'
                )
                
                if perm_importance_df is not None and not perm_importance_df.empty:
                    top_perm = perm_importance_df.head(15)
                    
                    fig = px.bar(
                        top_perm,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title=f"Top 15 Features (Permutation)",
                        color='importance',
                        color_continuous_scale='RdYlGn',
                        error_x='std' if 'std' in top_perm.columns else None
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
            
            if importance_df is not None and not importance_df.empty:
                with st.expander("📋 View Full Feature Importance Table"):
                    st.dataframe(importance_df, use_container_width=True)
        
        with analysis_tabs[1]:
            st.subheader(f"🎯 SHAP-Style Feature Contributions: {selected_model_name}")
            st.markdown("Using feature ablation to calculate contribution values (SHAP-style analysis)")
            
            with st.spinner("Calculating SHAP-style values..."):
                try:
                    shap_results = calculate_shap_style_values(
                        selected_model,
                        X_test,
                        feature_names,
                        sample_size=50
                    )
                    
                    shap_summary = get_shap_style_summary(shap_results, top_n=15)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        y=shap_summary['feature'],
                        x=shap_summary['mean_abs_shap'],
                        orientation='h',
                        name='Mean Absolute SHAP',
                        marker=dict(
                            color=shap_summary['mean_abs_shap'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        error_x=dict(type='data', array=shap_summary['std_shap'])
                    ))
                    
                    fig.update_layout(
                        title="SHAP-Style Feature Importance (Mean Absolute Contribution)",
                        xaxis_title="Mean Absolute SHAP Value",
                        yaxis_title="Feature",
                        yaxis={'categoryorder': 'total ascending'},
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Samples Analyzed", len(shap_results['shap_values']))
                    with col2:
                        st.metric("Features Analyzed", len(feature_names))
                    
                    with st.expander("📊 SHAP Summary Statistics"):
                        st.dataframe(shap_summary, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Feature Contribution Distribution")
                    
                    selected_feature = st.selectbox(
                        "Select feature to view distribution",
                        shap_summary['feature'].tolist()
                    )
                    
                    if selected_feature:
                        feature_shap = shap_results['shap_values'][selected_feature]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=feature_shap,
                            nbinsx=30,
                            name=selected_feature,
                            marker=dict(color='steelblue')
                        ))
                        fig.update_layout(
                            title=f"Distribution of SHAP Values: {selected_feature}",
                            xaxis_title="SHAP Value",
                            yaxis_title="Count",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write(f"**Statistics for {selected_feature}:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{feature_shap.mean():.4f}")
                        with col2:
                            st.metric("Std Dev", f"{feature_shap.std():.4f}")
                        with col3:
                            st.metric("Min", f"{feature_shap.min():.4f}")
                        with col4:
                            st.metric("Max", f"{feature_shap.max():.4f}")
                
                except Exception as e:
                    st.error(f"Error calculating SHAP-style values: {str(e)}")
                    st.info("This model may not support SHAP-style analysis")
        
        with analysis_tabs[2]:
            st.subheader(f"🔬 Individual Prediction Explanations")
            st.markdown("Waterfall-style visualization showing how features contribute to individual predictions")
            
            sample_idx = st.slider(
                "Select sample to explain",
                min_value=0,
                max_value=len(X_test)-1,
                value=0
            )
            
            try:
                shap_results = calculate_shap_style_values(
                    selected_model,
                    X_test,
                    feature_names,
                    sample_size=min(len(X_test), 100)
                )
                
                if sample_idx < len(shap_results['shap_values']):
                    waterfall_data = get_waterfall_data(shap_results, sample_idx)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Baseline Prediction", f"{waterfall_data['baseline']:.4f}")
                    with col2:
                        st.metric("Actual Prediction", f"{waterfall_data['prediction']:.4f}")
                    with col3:
                        delta = waterfall_data['prediction'] - waterfall_data['baseline']
                        st.metric("Difference", f"{delta:.4f}")
                    
                    st.markdown("---")
                    st.subheader("Top 10 Feature Contributions")
                    
                    contributions = waterfall_data['contributions']
                    
                    fig = go.Figure()
                    
                    colors = ['red' if c < 0 else 'green' for c in contributions['contribution']]
                    
                    fig.add_trace(go.Bar(
                        y=contributions['feature'],
                        x=contributions['contribution'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{c:.4f}" for c in contributions['contribution']],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Feature Contributions (Waterfall Style)",
                        xaxis_title="Contribution to Prediction",
                        yaxis_title="Feature",
                        yaxis={'categoryorder': 'total ascending'},
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📋 Detailed Contribution Data"):
                        contributions_display = contributions.copy()
                        contributions_display['feature_value'] = contributions_display['feature_value'].apply(lambda x: f"{x:.4f}")
                        contributions_display['contribution'] = contributions_display['contribution'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(contributions_display, use_container_width=True)
                else:
                    st.warning("Sample index out of range for SHAP analysis")
            
            except Exception as e:
                st.error(f"Error calculating individual prediction explanation: {str(e)}")
        
        with analysis_tabs[3]:
            st.subheader("📈 Partial Dependence Plots")
            st.markdown("Shows how predictions change as individual feature values change")
            
            if importance_df is not None and not importance_df.empty:
                top_3_features = importance_df.head(3)['feature'].tolist()
                
                selected_pd_feature = st.selectbox(
                    "Select feature for partial dependence",
                    top_3_features
                )
                
                try:
                    with st.spinner("Calculating partial dependence..."):
                        feature_values, pd_values = calculate_partial_dependence(
                            selected_model,
                            X_test,
                            selected_pd_feature,
                            num_points=30
                        )
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=feature_values,
                            y=pd_values,
                            mode='lines+markers',
                            name='Partial Dependence',
                            line=dict(color='steelblue', width=3)
                        ))
                        
                        fig.update_layout(
                            title=f"Partial Dependence: {selected_pd_feature}",
                            xaxis_title=f"{selected_pd_feature} Value",
                            yaxis_title="Average Prediction",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"This plot shows how the average prediction changes as **{selected_pd_feature}** varies while keeping other features at their mean values.")
                
                except Exception as e:
                    st.error(f"Error calculating partial dependence: {str(e)}")
    
    with tabs[4]:
        st.header("Lag Detection Analysis")
        
        st.markdown("This analysis identifies whether models are producing shifted predictions (lag) rather than true forecasts.")
        
        lag_detector = LagDetector(max_lag=10)
        
        y_test = data_split['y_test_reg']
        
        lag_results = {}
        for model_name, predictions in predictions_test.items():
            if task_type == 'classification':
                pred_continuous = predictions.astype(float)
            else:
                pred_continuous = predictions
            
            result = lag_detector.detect_lag(y_test.values, pred_continuous, model_name)
            lag_results[model_name] = result
        
        summary_df = lag_detector.get_lag_summary()
        
        if summary_df is not None:
            st.subheader("Lag Detection Summary")
            st.dataframe(summary_df, use_container_width=True)
            
            false_positives = lag_detector.identify_false_positives(threshold=0.05)
            
            if false_positives:
                st.warning("⚠️ Potential False Positives Detected")
                for fp in false_positives:
                    st.write(f"**{fp['model']}**: Lag of {fp['lag']} steps (Severity: {fp['severity']})")
            else:
                st.success("✅ No significant lag detected in predictions")
            
            st.markdown("---")
            st.subheader("Correlation Analysis")
            
            selected_lag_model = st.selectbox("Select Model for Correlation Plot", list(lag_results.keys()))
            
            if selected_lag_model in lag_results:
                correlations = lag_results[selected_lag_model]['all_correlations']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(correlations.keys()),
                    y=list(correlations.values()),
                    mode='lines+markers',
                    name='Correlation',
                    line=dict(color='blue', width=2)
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Lag")
                fig.update_layout(
                    title=f"Correlation at Different Lags: {selected_lag_model}",
                    xaxis_title="Lag (steps)",
                    yaxis_title="Correlation",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:
        st.header("Prediction Visualization")
        
        selected_viz_model = st.selectbox("Select Model for Prediction Visualization", list(models.keys()), key='viz_model')
        
        if task_type == 'regression':
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data_split['dates_test'],
                y=data_split['y_test_reg'],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data_split['dates_test'],
                y=predictions_test[selected_viz_model],
                mode='lines',
                name=f'{selected_viz_model} Prediction',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"Price Predictions: {selected_viz_model}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            y_test = data_split['y_test_class']
            predictions = predictions_test[selected_viz_model]
            
            results_df = pd.DataFrame({
                'Date': data_split['dates_test'],
                'Actual': ['Up' if y == 1 else 'Down' for y in y_test],
                'Predicted': ['Up' if p == 1 else 'Down' for p in predictions],
                'Correct': y_test.values == predictions
            })
            
            fig = go.Figure()
            
            correct_dates = results_df[results_df['Correct']]['Date']
            incorrect_dates = results_df[~results_df['Correct']]['Date']
            
            fig.add_trace(go.Scatter(
                x=correct_dates,
                y=[1] * len(correct_dates),
                mode='markers',
                name='Correct',
                marker=dict(color='green', size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=incorrect_dates,
                y=[0] * len(incorrect_dates),
                mode='markers',
                name='Incorrect',
                marker=dict(color='red', size=8)
            ))
            
            fig.update_layout(
                title=f"Prediction Accuracy: {selected_viz_model}",
                xaxis_title="Date",
                yaxis_title="",
                yaxis=dict(tickvals=[0, 1], ticktext=['Incorrect', 'Correct']),
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            accuracy = results_df['Correct'].mean()
            st.metric("Overall Accuracy", f"{accuracy:.2%}")

if __name__ == "__main__":
    main()
