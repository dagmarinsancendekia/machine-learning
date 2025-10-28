import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, filepath):
        """Load Excel data and return DataFrame"""
        try:
            df = pd.read_excel(filepath)
            return df
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def clean_data(self, df):
        """Basic data cleaning"""
        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values (simple imputation)
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

        return df

    def get_basic_stats(self, df):
        """Get basic statistics"""
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'summary': df.describe().to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': df.nunique().to_dict()
        }
        return stats

    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object', 'category']):
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
        return df_encoded

    def perform_regression(self, df, target_col):
        """Perform linear regression"""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical if any
        X = self.encode_categorical(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'model_type': 'Linear Regression',
            'mse': mse,
            'r2_score': r2,
            'feature_importance': dict(zip(X.columns, model.coef_))
        }

    def perform_clustering(self, df, n_clusters=3):
        """Perform K-means clustering"""
        # Encode categorical
        df_encoded = self.encode_categorical(df)

        # Scale data
        X_scaled = self.scaler.fit_transform(df_encoded)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        return {
            'model_type': 'K-means Clustering',
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_labels': clusters.tolist(),
            'inertia': kmeans.inertia_
        }

    def create_visualization(self, df, plot_type='correlation'):
        """Create data visualizations"""
        plt.figure(figsize=(10, 6))

        if plot_type == 'correlation':
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
            else:
                plt.text(0.5, 0.5, 'No numeric columns for correlation', ha='center')

        elif plot_type == 'histogram':
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # First 4 numeric columns
            if len(numeric_cols) > 0:
                df[numeric_cols].hist(bins=30, figsize=(12, 8))
                plt.tight_layout()
            else:
                plt.text(0.5, 0.5, 'No numeric columns for histogram', ha='center')

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"
