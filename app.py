import os
import glob
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Import inference class
try:
    from inference import ImageRetriever
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference import ImageRetriever

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize Model (Load once at startup)
CHECKPOINT_PATH = "experiments/runs/20260420-115433/best_model.pth"
CONFIG_PATH = "configs/et_edu_config.yaml"
DATABASE_DIR = "data/cropped_students_quality_10k"

print("Initializing G-Hash Image Retriever...")
try:
    retriever = ImageRetriever(checkpoint_path=CHECKPOINT_PATH, config_path=CONFIG_PATH)
    
    # Pre-build database index
    print(f"Loading database from train_img.txt and test_img.txt...")
    database_images = []
    db_root = Path("data")
    for txt_file in ["train_img.txt", "test_img.txt"]:
        txt_path = db_root / txt_file
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    img_path = str(db_root / line.strip())
                    database_images.append(img_path)
                    
    print(f"Found {len(database_images)} images. Building index...")
    if database_images:
        retriever.build_database_index(database_images)
        print("Database index built successfully.")
    else:
        print("Warning: No images found in train_img.txt or test_img.txt")
except Exception as e:
    print(f"Error initializing model: {e}")
    retriever = None
    database_images = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if retriever is None:
        return jsonify({'error': 'Model not initialized properly.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Predict Labels
            predictions = retriever.predict_labels(filepath, top_k=5)
            
            # 2. Retrieve Similar Images
            retrieval_results = []
            if len(database_images) > 0:
                results = retriever.retrieve_similar_images(filepath, database_images, top_k=10)
                for res in results:
                    # Return relative path for serving
                    img_path = res['image_path']
                    # Map local absolute/relative path to a URL path
                    retrieval_results.append({
                        'rank': res['rank'],
                        'image_url': '/' + img_path.replace('\\', '/'),
                        'hamming_distance': res['hamming_distance'],
                        'rerank_score': res['rerank_score']
                    })
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'retrieval': retrieval_results
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Unknown error'}), 500

# Route to serve database images
@app.route('/data/<path:filename>')
def serve_data_image(filename):
    return send_from_directory('data', filename)

# Route to serve upload images
@app.route('/uploads/<path:filename>')
def serve_upload_image(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
