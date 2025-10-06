import os
import pandas as pd
import numpy as np
import json
import base64
import io
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from models.advanced_anomaly_detector import AdvancedAnomalyDetector
from preprocessing.extract_features import extract_features
from utils.data_merger import DataMerger
from utils.date_parser import FlexibleDateParser
from utils.auth import auth_store
from config import Config
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = Config.SEND_FILE_MAX_AGE_DEFAULT

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global variables to store model and results
model = None
current_results = None
data_merger = DataMerger()
date_parser = FlexibleDateParser()

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_uploaded_data(file_paths, upload_mode='single'):
    """Preprocess uploaded CSV data"""
    try:
        if upload_mode == 'single':
            # Single file mode - check if it's a large file
            if Config.is_large_file(file_paths):
                file_size_gb = Config.get_file_size_gb(file_paths)
                print(f"Large file detected ({file_size_gb:.2f} GB). Using chunked processing...")
                from utils.chunked_processor import ChunkedProcessor
                processor = ChunkedProcessor(
                    chunk_size=Config.CHUNK_SIZE,
                    max_memory_usage=Config.MAX_MEMORY_USAGE
                )
                df = processor.extract_features_chunked(file_paths)
            else:
                df = pd.read_csv(file_paths)
            
            # Check for essential columns (user and date are mandatory)
            essential_columns = ['user', 'date']
            missing_essential = [col for col in essential_columns if col not in df.columns]
            
            if missing_essential:
                return None, f"Missing essential columns: {', '.join(missing_essential)}"
            
            # Parse dates flexibly
            if 'date' in df.columns:
                original_count = len(df)
                df['date'] = date_parser.parse_date_column(df['date'])
                # Remove rows with invalid dates (skip problematic data)
                df = df.dropna(subset=['date'])
                skipped_count = original_count - len(df)
                
                if skipped_count > 0:
                    print(f"Info: Skipped {skipped_count} rows with invalid dates")
                
                if len(df) == 0:
                    return None, "No valid dates found in the file - all rows were skipped due to date format issues"
            
            # Ensure all expected activity columns exist, add defaults if missing
            expected_activity_columns = ['logon', 'file', 'email', 'device', 'http']
            for col in expected_activity_columns:
                if col not in df.columns:
                    print(f"Warning: {col} column missing, using default value 0")
                    df[col] = 0
        
        elif upload_mode == 'multiple':
            # Multiple files mode - merge separate files
            merged_df, error = data_merger.merge_files(file_paths)
            if error:
                return None, error
            df = merged_df
        
        # Extract features using the existing preprocessing pipeline
        features_df = extract_features(df)
        
        return features_df, None
        
    except Exception as e:
        return None, f"Error preprocessing data: {str(e)}"

def create_visualizations(results_df, output_dir='static'):
    """Create visualization plots for the results"""
    plots = {}
    
    try:
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Score distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(results_df[results_df['pseudo_label'] == 0]['advanced_score'], 
                bins=30, alpha=0.7, label='Normal', color='blue')
        plt.hist(results_df[results_df['pseudo_label'] == 1]['advanced_score'], 
                bins=30, alpha=0.7, label='Anomaly', color='red')
        plt.xlabel('Advanced Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Top anomalies bar chart
        plt.subplot(2, 2, 2)
        top_anomalies = results_df[results_df['pseudo_label'] == 1].head(10)
        plt.barh(range(len(top_anomalies)), top_anomalies['advanced_score'])
        plt.yticks(range(len(top_anomalies)), top_anomalies['user'])
        plt.xlabel('Anomaly Score')
        plt.title('Top 10 Anomalous Users')
        plt.grid(True, alpha=0.3)
        
        # Determine numeric feature columns only, excluding identifier/label columns
        exclude_cols = ['user', 'advanced_score', 'pseudo_label']
        numeric_cols = [c for c in results_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        # 3. Feature correlation heatmap (only if we have >= 2 numeric features)
        plt.subplot(2, 2, 3)
        if len(numeric_cols) >= 2:
            corr_matrix = results_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix')
        else:
            plt.axis('off')
            plt.text(0.5, 0.5, 'Not enough numeric features for correlation heatmap', ha='center', va='center')
        
        # 4. Anomaly vs Normal feature comparison (if we have >= 1 numeric feature)
        plt.subplot(2, 2, 4)
        if len(numeric_cols) >= 1:
            normal_features = results_df[results_df['pseudo_label'] == 0][numeric_cols].mean()
            anomaly_features = results_df[results_df['pseudo_label'] == 1][numeric_cols].mean()
            
            x = np.arange(len(numeric_cols))
            width = 0.35
            
            plt.bar(x - width/2, normal_features, width, label='Normal', alpha=0.8)
            plt.bar(x + width/2, anomaly_features, width, label='Anomaly', alpha=0.8)
            plt.xlabel('Features')
            plt.ylabel('Average Values')
            plt.title('Feature Comparison: Normal vs Anomaly')
            plt.xticks(x, numeric_cols, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.axis('off')
            plt.text(0.5, 0.5, 'No numeric features to compare', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plots['analysis_plots'] = 'analysis_plots.png'
        
        # 5. Individual user analysis for top anomalies
        if len(top_anomalies) > 0:
            plt.figure(figsize=(15, 10))
            
            for i, (_, user_data) in enumerate(top_anomalies.head(6).iterrows()):
                plt.subplot(2, 3, i+1)
                
                # Create radar chart for user features if we have enough numeric features
                if len(numeric_cols) >= 3:
                    user_features = user_data[numeric_cols].values
                    # Normalize features for radar chart
                    min_vals = results_df[numeric_cols].min()
                    max_vals = results_df[numeric_cols].max()
                    denom = (max_vals - min_vals).replace(0, 1)
                    user_features_norm = (user_features - min_vals.values) / denom.values
                    
                    angles = np.linspace(0, 2 * np.pi, len(numeric_cols), endpoint=False).tolist()
                    user_features_norm = np.concatenate((user_features_norm, [user_features_norm[0]]))
                    angles += angles[:1]
                    
                    plt.polar(angles, user_features_norm, 'o-', linewidth=2, label=user_data['user'])
                    plt.fill(angles, user_features_norm, alpha=0.25)
                    plt.xticks(angles[:-1], numeric_cols, fontsize=8)
                    plt.ylim(0, 1)
                    plt.title(f"User: {user_data['user']}\nScore: {user_data['advanced_score']:.3f}", fontsize=10)
                    plt.grid(True)
                else:
                    plt.axis('off')
                    plt.text(0.5, 0.5, 'Not enough features for radar chart', ha='center', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/user_radar_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            plots['user_radar_charts'] = 'user_radar_charts.png'
        
        return plots
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        return {}

@app.route('/')
def index():
    """Main page (requires login)"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/healthz')
def healthz():
    return jsonify({'ok': True}), 200

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        error = auth_store.create_user(username, password)
        if error:
            return render_template('signup.html', error=error, username=username)
        session['username'] = username.lower()
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if auth_store.verify_user(username, password):
            session['username'] = username.lower()
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid username or password', username=username)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    global model, current_results
    
    try:
        upload_mode = request.form.get('upload_mode', 'single')
        
        if upload_mode == 'single':
            # Single file upload (original functionality)
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Check file size and provide feedback
                file_size_gb = Config.get_file_size_gb(file_path)
                
                if file_size_gb > 1:  # > 1GB
                    print(f"Large file uploaded: {file_size_gb:.2f} GB")
                
                # Preprocess the data
                features_df, error = preprocess_uploaded_data(file_path, 'single')
                if error:
                    return jsonify({'error': error}), 400
                
                return process_data(features_df)
            else:
                return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
        elif upload_mode == 'multiple':
            # Multiple files upload
            file_types = ['logon', 'file', 'email', 'device', 'http']
            file_paths = {}
            total_size = 0
            
            for file_type in file_types:
                if file_type in request.files:
                    file = request.files[file_type]
                    if file and file.filename and allowed_file(file.filename):
                        filename = secure_filename(f"{file_type}_{file.filename}")
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        file_paths[file_type] = file_path
                        total_size += os.path.getsize(file_path)
            
            if not file_paths:
                return jsonify({'error': 'No valid files uploaded'}), 400
            
            # Check total size
            total_size_gb = total_size / (1024 * 1024 * 1024)
            if total_size_gb > 1:  # > 1GB
                print(f"Large files uploaded: {total_size_gb:.2f} GB total")
            
            # Validate each file
            for file_type, file_path in file_paths.items():
                is_valid, error_msg = data_merger.validate_file(file_path, file_type)
                if not is_valid:
                    return jsonify({'error': error_msg}), 400
            
            # Preprocess the merged data
            features_df, error = preprocess_uploaded_data(file_paths, 'multiple')
            if error:
                return jsonify({'error': error}), 400
            
            return process_data(features_df)
        
        else:
            return jsonify({'error': 'Invalid upload mode'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

def process_data(features_df):
    """Process the features dataframe and return results"""
    global model, current_results
    
    try:
        # Initialize and train the model
        X = features_df.drop(columns=['user'])
        model = AdvancedAnomalyDetector(
            device='cpu',
            autoencoder_params={
                'encoding_dim': Config.AUTOENCODER_ENCODING_DIM,
                'dropout_rate': Config.AUTOENCODER_DROPOUT_RATE
            }
        )
        
        # Train the model
        model.fit(X)
        
        # Get predictions
        advanced_scores = model.predict(X)
        features_df['advanced_score'] = advanced_scores
        
        # Mark top 5% as anomalies
        thresh = np.percentile(advanced_scores, Config.ANOMALY_THRESHOLD_PERCENTILE)
        features_df['pseudo_label'] = (advanced_scores > thresh).astype(int)
        
        # Store results
        current_results = features_df
        
        # Create visualizations
        plots = create_visualizations(features_df)
        
        # Prepare summary statistics
        total_users = len(features_df)
        anomalies_detected = len(features_df[features_df['pseudo_label'] == 1])
        anomaly_rate = (anomalies_detected / total_users) * 100
        
        # Get top anomalies
        top_anomalies = features_df[features_df['pseudo_label'] == 1].sort_values('advanced_score', ascending=False).head(10)
        
        return jsonify({
            'success': True,
            'summary': {
                'total_users': total_users,
                'anomalies_detected': anomalies_detected,
                'anomaly_rate': round(anomaly_rate, 2),
                'threshold': round(thresh, 3)
            },
            'top_anomalies': top_anomalies[['user', 'advanced_score']].to_dict('records'),
            'plots': plots
        })
        
    except Exception as e:
        return jsonify({'error': f'Model processing error: {str(e)}'}), 500

@app.route('/results')
def get_results():
    """Get detailed results"""
    global current_results
    
    if current_results is None:
        return jsonify({'error': 'No results available'}), 400
    
    # Convert results to JSON-serializable format
    results_data = current_results.to_dict('records')
    
    return jsonify({
        'success': True,
        'data': results_data,
        'columns': list(current_results.columns)
    })

@app.route('/download_results')
def download_results():
    """Download results as CSV"""
    global current_results
    
    if current_results is None:
        return jsonify({'error': 'No results available'}), 400
    
    # Create CSV in memory
    output = io.StringIO()
    current_results.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='anomaly_detection_results.csv'
    )

@app.route('/user_details/<user_id>')
def get_user_details(user_id):
    """Get detailed information for a specific user"""
    global current_results
    
    if current_results is None:
        return jsonify({'error': 'No results available'}), 400
    
    user_data = current_results[current_results['user'] == user_id]
    
    if user_data.empty:
        return jsonify({'error': 'User not found'}), 404
    
    user_info = user_data.iloc[0].to_dict()
    
    # Calculate percentiles
    feature_cols = [col for col in current_results.columns if col not in ['user', 'advanced_score', 'pseudo_label']]
    percentiles = {}
    
    for col in feature_cols:
        user_value = user_info[col]
        percentile = (current_results[col] <= user_value).mean() * 100
        percentiles[col] = round(percentile, 2)
    
    return jsonify({
        'success': True,
        'user_info': user_info,
        'percentiles': percentiles,
        'feature_columns': feature_cols
    })

@app.route('/create_sample_files')
def create_sample_files():
    """Create sample separate CSV files for testing"""
    try:
        message = data_merger.create_sample_files('sample_data')
        return jsonify({'success': True, 'message': message})
    except Exception as e:
        return jsonify({'error': f'Error creating sample files: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
