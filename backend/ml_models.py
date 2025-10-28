import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class MLModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_data(self, df, target_col, test_size=0.2):
        """Prepare data for training"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Ensure we have at least 2 samples for splitting
        if len(df) < 2:
            raise ValueError("Dataset must have at least 2 samples")

        # Adjust test_size if dataset is too small
        if len(df) < 5:
            test_size = 0.5
        elif len(df) < 10:
            test_size = 0.3

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_regression_model(self, df, target_col, model_type='linear'):
        """Train regression model"""
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVR(kernel='rbf')
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Save model
        model_path = os.path.join(self.model_dir, f'regression_{model_type}_{target_col}.pkl')
        joblib.dump(model, model_path)

        return {
            'model_type': model_type,
            'target': target_col,
            'metrics': metrics,
            'model_path': model_path,
            'feature_importance': getattr(model, 'feature_importances_', None)
        }

    def train_classification_model(self, df, target_col, model_type='logistic'):
        """Train classification model"""
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        if model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            model = SVC(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        # Save model
        model_path = os.path.join(self.model_dir, f'classification_{model_type}_{target_col}.pkl')
        joblib.dump(model, model_path)

        return {
            'model_type': model_type,
            'target': target_col,
            'metrics': metrics,
            'model_path': model_path,
            'feature_importance': getattr(model, 'feature_importances_', None)
        }

    def predict(self, model_path, input_data):
        """Make predictions using saved model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        # Prepare input data (similar to training preparation)
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        # Ensure columns match training data (this is a simplified version)
        # In production, you'd save the column order during training

        predictions = model.predict(input_df)
        return predictions.tolist()

    def get_available_models(self):
        """List all saved models"""
        if not os.path.exists(self.model_dir):
            return []

        models = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl'):
                models.append(file)
        return models
