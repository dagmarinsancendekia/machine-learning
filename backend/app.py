from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from werkzeug.utils import secure_filename
import json
from data_processor import DataProcessor
from ml_models import MLModelTrainer

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Initialize components
data_processor = DataProcessor()
model_trainer = MLModelTrainer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Excel Analyzer API is running'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Basic file validation and info extraction
            df = pd.read_excel(filepath)
            file_info = {
                'filename': filename,
                'filepath': filepath,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            return jsonify({'message': 'File uploaded successfully', 'file_info': file_info}), 200
        except Exception as e:
            os.remove(filepath)  # Clean up on error
            return jsonify({'error': f'Error processing file: {str(e)}'}), 400

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_excel(filepath)

        # Use data processor for comprehensive analysis
        stats = data_processor.get_basic_stats(df)

        return jsonify({'message': 'Analysis completed', 'analysis': stats}), 200
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/clean-data', methods=['POST'])
def clean_data():
    data = request.get_json()
    filepath = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_excel(filepath)
        cleaned_df = data_processor.clean_data(df)

        # Save cleaned data
        cleaned_filepath = filepath.replace('.xlsx', '_cleaned.xlsx').replace('.xls', '_cleaned.xls')
        cleaned_df.to_excel(cleaned_filepath, index=False)

        return jsonify({
            'message': 'Data cleaned successfully',
            'original_shape': df.shape,
            'cleaned_shape': cleaned_df.shape,
            'cleaned_filepath': cleaned_filepath
        }), 200
    except Exception as e:
        return jsonify({'error': f'Data cleaning failed: {str(e)}'}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    data = request.get_json()
    filepath = data.get('filepath')
    target_col = data.get('target_column')
    model_type = data.get('model_type', 'regression')
    algorithm = data.get('algorithm', 'linear')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    if not target_col:
        return jsonify({'error': 'Target column is required'}), 400

    try:
        df = pd.read_excel(filepath)

        if model_type == 'regression':
            result = model_trainer.train_regression_model(df, target_col, algorithm)
        elif model_type == 'classification':
            result = model_trainer.train_classification_model(df, target_col, algorithm)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        return jsonify({'message': 'Model trained successfully', 'result': result}), 200
    except Exception as e:
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

@app.route('/api/visualize', methods=['POST'])
def create_visualization():
    data = request.get_json()
    filepath = data.get('filepath')
    plot_type = data.get('plot_type', 'correlation')

    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        df = pd.read_excel(filepath)
        image_data = data_processor.create_visualization(df, plot_type)

        return jsonify({
            'message': 'Visualization created',
            'image': image_data,
            'plot_type': plot_type
        }), 200
    except Exception as e:
        return jsonify({'error': f'Visualization failed: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        models = model_trainer.get_available_models()
        return jsonify({'models': models}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve models: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
