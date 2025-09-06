from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
from werkzeug.utils import secure_filename
import logging
from improved_asl_recognition import ImprovedASLRecognizer
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = '/home/ubuntu/asl-translator/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize improved ASL recognizer
recognizer = ImprovedASLRecognizer()

# Store processing results
processing_results = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_async(video_path, task_id):
    """Process video asynchronously"""
    try:
        logger.info(f"Starting video processing for task {task_id}")
        
        # Update status
        processing_results[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing video processing...'
        }
        
        # Process the video with enhanced method
        results = recognizer.process_video_enhanced(video_path)
        
        if results['success']:
            # Get additional statistics
            stats = recognizer.get_recognition_stats(results)
            
            processing_results[task_id] = {
                'status': 'completed',
                'progress': 100,
                'message': 'Video processing completed successfully',
                'results': results,
                'stats': stats
            }
            
            logger.info(f"Video processing completed for task {task_id}")
        else:
            processing_results[task_id] = {
                'status': 'error',
                'progress': 0,
                'message': f"Processing failed: {results.get('error', 'Unknown error')}",
                'error': results.get('error', 'Unknown error')
            }
            
    except Exception as e:
        logger.error(f"Error processing video for task {task_id}: {e}")
        processing_results[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f"Processing failed: {str(e)}",
            'error': str(e)
        }
    
    finally:
        # Clean up video file after processing
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up video file: {video_path}")
        except Exception as e:
            logger.warning(f"Could not clean up video file {video_path}: {e}")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload MP4, AVI, MOV, MKV, or WEBM files.'}), 400
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{task_id}.{file_extension}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(video_path)
        logger.info(f"Video uploaded: {video_path}")
        
        # Start processing in background thread
        thread = threading.Thread(target=process_video_async, args=(video_path, task_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Video uploaded successfully. Processing started.'
        })
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Get processing status for a task"""
    try:
        if task_id not in processing_results:
            return jsonify({'error': 'Task not found'}), 404
        
        result = processing_results[task_id]
        
        # Clean up completed/error tasks after 1 hour
        if result['status'] in ['completed', 'error']:
            # You might want to implement cleanup logic here
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {e}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.route('/result/<task_id>')
def get_result(task_id):
    """Get detailed results for a completed task"""
    try:
        if task_id not in processing_results:
            return jsonify({'error': 'Task not found'}), 404
        
        result = processing_results[task_id]
        
        if result['status'] != 'completed':
            return jsonify({'error': 'Task not completed yet'}), 400
        
        return jsonify({
            'success': True,
            'text_output': result['results']['text_output'],
            'detected_letters': result['results']['detected_letters'],
            'frame_results': result['results']['frame_results'],
            'video_info': result['results']['video_info'],
            'stats': result['stats'],
            'confidence_avg': result['results']['confidence_avg']
        })
        
    except Exception as e:
        logger.error(f"Error getting result for task {task_id}: {e}")
        return jsonify({'error': f'Result retrieval failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'recognizer_initialized': recognizer is not None,
        'upload_folder': os.path.exists(UPLOAD_FOLDER),
        'active_tasks': len(processing_results)
    })

@app.route('/demo')
def demo():
    """Demo endpoint that returns sample recognition results"""
    sample_result = {
        'success': True,
        'text_output': 'HELLO WORLD THANK YOU',
        'detected_letters': ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D', 'T', 'H', 'A', 'N', 'K', 'Y', 'O', 'U'],
        'confidence_avg': 0.87,
        'video_info': {
            'duration': 5.2,
            'fps': 30,
            'total_frames': 156,
            'processed_frames': 31
        },
        'stats': {
            'total_detections': 18,
            'unique_letters': 12,
            'avg_confidence': 0.87,
            'processing_success_rate': 58.1
        }
    }
    
    return jsonify(sample_result)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    logger.info("Starting ASL Translator Flask application...")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
    
    # Test the recognizer
    try:
        test_landmarks = recognizer.extract_hand_landmarks.__doc__
        logger.info("ASL Recognizer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ASL Recognizer: {e}")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

