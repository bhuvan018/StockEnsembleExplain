import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class CNNModel:
    def __init__(self, input_shape, task='classification', lookback=10):
        self.name = "CNN"
        self.task = task
        self.lookback = lookback
        self.input_shape = input_shape
        self.model = None
        self.fitted = False
        self.history = None
        
    def build_model(self):
        if self.task == 'classification':
            self.model = models.Sequential([
                layers.Reshape((self.lookback, -1, 1), input_shape=(self.input_shape,)),
                layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 1)),
                layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 1)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.model = models.Sequential([
                layers.Reshape((self.lookback, -1, 1), input_shape=(self.input_shape,)),
                layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 1)),
                layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 1)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1)
            ])
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
    
    def prepare_sequences(self, X):
        n_features = X.shape[1] // self.lookback
        X_reshaped = X.values.reshape(X.shape[0], self.lookback, n_features)
        return X_reshaped
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0):
        if self.model is None:
            self.build_model()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.fitted = True
    
    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X_test, verbose=0)
        
        if self.task == 'classification':
            return (predictions > 0.5).astype(int).flatten()
        else:
            return predictions.flatten()
    
    def predict_proba(self, X_test):
        if not self.fitted or self.task != 'classification':
            return None
        
        predictions = self.model.predict(X_test, verbose=0)
        proba = np.hstack([1 - predictions, predictions])
        return proba
    
    def get_params(self):
        return {
            'model_type': 'Deep Learning',
            'algorithm': 'CNN',
            'task': self.task,
            'parameters': {
                'lookback': self.lookback,
                'input_shape': self.input_shape,
                'total_params': self.model.count_params() if self.model else 0
            }
        }


class GRUModel:
    def __init__(self, input_shape, task='classification', lookback=10):
        self.name = "GRU"
        self.task = task
        self.lookback = lookback
        self.input_shape = input_shape
        self.model = None
        self.fitted = False
        self.history = None
        
    def build_model(self):
        n_features = self.input_shape // self.lookback
        
        if self.task == 'classification':
            self.model = models.Sequential([
                layers.GRU(64, return_sequences=True, input_shape=(self.lookback, n_features)),
                layers.Dropout(0.2),
                layers.GRU(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.model = models.Sequential([
                layers.GRU(64, return_sequences=True, input_shape=(self.lookback, n_features)),
                layers.Dropout(0.2),
                layers.GRU(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
    
    def prepare_sequences(self, X):
        n_features = X.shape[1] // self.lookback
        X_reshaped = X.values.reshape(X.shape[0], self.lookback, n_features)
        return X_reshaped
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0):
        if self.model is None:
            self.build_model()
        
        X_train_seq = self.prepare_sequences(X_train)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train_seq, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.fitted = True
    
    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_seq = self.prepare_sequences(X_test)
        predictions = self.model.predict(X_test_seq, verbose=0)
        
        if self.task == 'classification':
            return (predictions > 0.5).astype(int).flatten()
        else:
            return predictions.flatten()
    
    def predict_proba(self, X_test):
        if not self.fitted or self.task != 'classification':
            return None
        
        X_test_seq = self.prepare_sequences(X_test)
        predictions = self.model.predict(X_test_seq, verbose=0)
        proba = np.hstack([1 - predictions, predictions])
        return proba
    
    def get_params(self):
        return {
            'model_type': 'Deep Learning',
            'algorithm': 'GRU',
            'task': self.task,
            'parameters': {
                'lookback': self.lookback,
                'input_shape': self.input_shape,
                'total_params': self.model.count_params() if self.model else 0
            }
        }


class LSTMModel:
    def __init__(self, input_shape, task='classification', lookback=10):
        self.name = "LSTM"
        self.task = task
        self.lookback = lookback
        self.input_shape = input_shape
        self.model = None
        self.fitted = False
        self.history = None
        
    def build_model(self):
        n_features = self.input_shape // self.lookback
        
        if self.task == 'classification':
            self.model = models.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(self.lookback, n_features)),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            self.model = models.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(self.lookback, n_features)),
                layers.Dropout(0.2),
                layers.LSTM(32),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
    
    def prepare_sequences(self, X):
        n_features = X.shape[1] // self.lookback
        X_reshaped = X.values.reshape(X.shape[0], self.lookback, n_features)
        return X_reshaped
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0):
        if self.model is None:
            self.build_model()
        
        X_train_seq = self.prepare_sequences(X_train)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train_seq, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.fitted = True
    
    def predict(self, X_test):
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_test_seq = self.prepare_sequences(X_test)
        predictions = self.model.predict(X_test_seq, verbose=0)
        
        if self.task == 'classification':
            return (predictions > 0.5).astype(int).flatten()
        else:
            return predictions.flatten()
    
    def predict_proba(self, X_test):
        if not self.fitted or self.task != 'classification':
            return None
        
        X_test_seq = self.prepare_sequences(X_test)
        predictions = self.model.predict(X_test_seq, verbose=0)
        proba = np.hstack([1 - predictions, predictions])
        return proba
    
    def get_params(self):
        return {
            'model_type': 'Deep Learning',
            'algorithm': 'LSTM',
            'task': self.task,
            'parameters': {
                'lookback': self.lookback,
                'input_shape': self.input_shape,
                'total_params': self.model.count_params() if self.model else 0
            }
        }
