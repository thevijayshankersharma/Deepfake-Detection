from flask import Flask, render_template, redirect, request, url_for, send_file, send_from_directory, flash
from flask import jsonify, json
from werkzeug.utils import secure_filename
import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, SecurityAlert
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import face_recognition
import librosa
from torch.autograd import Variable
import time
import uuid
import sys
import traceback
import logging
import zipfile
from torch import nn
import torch.nn.functional as F
from torchvision import models
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from urllib.parse import urlparse, urlunparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path for the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Uploaded_Files')
FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
GRAPHS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'graphs')
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Admin', 'datasets')

# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Ensure folders have proper permissions
os.chmod(FRAMES_FOLDER, 0o755)
os.chmod(GRAPHS_FOLDER, 0o755)
os.chmod(DATASET_FOLDER, 0o755)

video_path = ""
detectOutput = []

app = Flask("__main__", template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize SQLAlchemy
db.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create all database tables
with app.app_context():
    db.create_all()

# Dataset comparison accuracies
DATASET_ACCURACIES = {
    'Our Model (ViT)': 92,  #  Replace with your ViT model's accuracy
    'FaceForensics++': 85.1,
    'DeepFake Detection Challenge': 82.3,
    'DeeperForensics-1.0': 80.7,
    'Previous Model (ResNeXt+LSTM)': 96  # Keep the old model for comparison
}

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        user = User.query.filter_by(email=email).first()
        if user:
            return render_template('signup.html', error="Email already exists")

        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('signup.html', error="Username already exists")

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('homepage'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('homepage'))
        else:
            return render_template('login.html', error="Invalid email or password")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('homepage'))

def generate_confidence_graph(confidence):
    try:
        plt.figure(figsize=(10, 10))
        plt.style.use('dark_background')

        real_cmap = LinearSegmentedColormap.from_list('custom_real', ['#2ecc71', '#27ae60'])
        fake_cmap = LinearSegmentedColormap.from_list('custom_fake', ['#ff0000', '#cc0000'])

        colors = [real_cmap(0.6), fake_cmap(0.9)]

        sizes = [confidence, 100 - confidence]
        labels = ['Real', 'Fake']
        explode = (0, 0.1)

        wedges, texts, autotexts = plt.pie(sizes,
                                          explode=explode,
                                          labels=labels,
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          shadow=True,
                                          startangle=90,
                                          textprops={'fontsize': 14, 'color': 'white'},
                                          wedgeprops={'edgecolor': '#2c3e50', 'linewidth': 2})

        plt.setp(autotexts, size=12, weight="bold")
        plt.setp(texts, size=14, weight="bold")

        plt.title('Confidence Score',
                 pad=20,
                 fontsize=16,
                 fontweight='bold',
                 color='white')

        plt.axis('equal')
        plt.grid(True, alpha=0.1, linestyle='--')

        unique_id = str(uuid.uuid4()).split('-')[0]
        graph_filename = f'confidence_{unique_id}.png'
        graph_path = os.path.join(GRAPHS_FOLDER, graph_filename)
        plt.savefig(graph_path,
                   bbox_inches='tight',
                   dpi=300,
                   transparent=True,
                   facecolor='#1a1a1a')
        plt.close()

        logger.info(f"Generated confidence graph: {graph_filename}")
        return f'graphs/{graph_filename}'
    except Exception as e:
        logger.error(f"Error generating confidence graph: {str(e)}")
        traceback.print_exc()
        return None

def generate_comparison_graph(our_accuracy):
    try:
        plt.figure(figsize=(14, 8))  # Increased figure size for more labels
        plt.style.use('dark_background')

        DATASET_ACCURACIES['Our Model (ViT)'] = our_accuracy

        datasets = list(DATASET_ACCURACIES.keys())
        accuracies = list(DATASET_ACCURACIES.values())

        main_color = '#64ffda'
        secondary_colors = ['#34495e', '#2c3e50', '#2980b9', '#f44336']  # Added more colors
        colors = [main_color] + secondary_colors[:len(datasets) - 1]

        plt.gca().set_facecolor('#111d40')
        plt.gcf().set_facecolor('#111d40')

        bars = plt.bar(datasets, accuracies, color=colors)

        plt.grid(axis='y', linestyle='--', alpha=0.2, color='white')

        plt.title('Model Performance Comparison',
                 color='white',
                 pad=20,
                 fontsize=16,
                 fontweight='bold')

        plt.xlabel('Models',
                  color='white',
                  labelpad=10,
                  fontsize=12,
                  fontweight='bold')

        plt.ylabel('Accuracy (%)',
                  color='white',
                  labelpad=10,
                  fontsize=12,
                  fontweight='bold')

        plt.xticks(rotation=45,  # Increased rotation for better readability
                   ha='right',
                   color='#8892b0',
                   fontsize=10)

        plt.yticks(color='#8892b0',
                   fontsize=10)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center',
                     va='bottom',
                     color='white',
                     fontsize=10,  # Slightly reduced font size for more labels
                     fontweight='bold',
                     bbox=dict(facecolor='#111d40',
                               edgecolor='none',
                               alpha=0.7,
                               pad=3))

        plt.box(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#34495e')
        plt.gca().spines['bottom'].set_color('#34495e')

        plt.tight_layout()
        unique_id = str(uuid.uuid4()).split('-')[0]
        graph_filename = f'comparison_{unique_id}.png'
        graph_path = os.path.join(GRAPHS_FOLDER, graph_filename)

        plt.savefig(graph_path,
                    bbox_inches='tight',
                    dpi=300,
                    transparent=True,
                    facecolor='#111d40')
        plt.close()

        logger.info(f"Generated comparison graph: {graph_filename}")
        return f'graphs/{graph_filename}'
    except Exception as e:
        logger.error(f"Error generating comparison graph: {str(e)}")
        traceback.print_exc()
        return None

# --- Define your ViT Model ---
class ViTModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTModel, self).__init__()
        #  ViT model architecture matching 'ashish-001/deepfake-detection-using-ViT'
        self.vit = ViTForImageClassification.from_pretrained("ashish-001/deepfake-detection-using-ViT")
        self.fc = nn.Linear(self.vit.classifier.in_features, num_classes)
        self.vit.classifier = self.fc # Replace the classifier

    def forward(self, x):
        x = self.vit(x).logits
        return None, x #fmap, logits

def extract_frames(video_path, num_frames=8):
    frames = []
    frame_paths = []
    unique_id = str(uuid.uuid4()).split('-')[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise Exception("Video file appears to be empty")

    interval = total_frames // num_frames

    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0 and frame_count < num_frames:
            faces = face_recognition.face_locations(frame)
            if len(faces) == 0:
                continue

            try:
                top, right, bottom, left = faces[0]
                face_frame = frame[top:bottom, left:right, :]
                frame_path = os.path.join(FRAMES_FOLDER, f'frame_{unique_id}_{frame_count}.jpg')
                cv2.imwrite(frame_path, face_frame)
                frame_paths.append(os.path.basename(frame_path))
                frames.append(face_frame)
                frame_count += 1
                logger.info(f"Extracted frame {frame_count}: {os.path.basename(frame_path)}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                continue

        count += 1
        if frame_count >= num_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise Exception("No faces detected in the video")

    return frames, frame_paths

def predict_vit(model, img):
    try:
        with torch.no_grad():
            outputs = model(img)  # Get the output from the model
            logits = outputs.logits  # Extract logits from ImageClassifierOutput
            probabilities = F.softmax(logits, dim=1)
            # Average probabilities across all frames
            avg_probabilities = probabilities.mean(dim=0)
            confidence, predicted_class = torch.max(avg_probabilities, 0)
            confidence_value = confidence.item() * 100
            logger.info(f'Prediction confidence: {confidence_value}%')
            return [int(predicted_class.item()), confidence_value]
    except Exception as e:
        logger.error(f"Error during ViT prediction: {str(e)}")
        traceback.print_exc()
        raise

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frame = Image.fromarray(frame)  # Convert to PIL Image
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            if (len(frames) == self.count):
                break
        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames)
        return frames_tensor

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def detectFakeVideo(videoPath):
    start_time = time.time()

    try:
        # Load ViT model
        vit_model = ViTForImageClassification.from_pretrained("ashish-001/deepfake-detection-using-ViT")  # Load the model
        vit_model.eval()  # Set to evaluation mode

        # Preprocessing
        im_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        path_to_videos = [videoPath]
        video_dataset = validation_dataset(path_to_videos, sequence_length=8, transform=transform)
        frames = video_dataset[0]  # Get the tensor of frames
        if frames.shape[0] == 0:
            raise Exception("No frames could be extracted from the video")
        
        prediction = predict_vit(vit_model, frames)  # Pass the tensor to predict_vit
        processing_time = time.time() - start_time

        # Create security alert if deepfake is detected
        if prediction[0] == 0 and current_user.is_authenticated:  # 0 indicates fake
            create_security_alert(
                user_id=current_user.id,
                alert_type='deepfake',
                severity='high',
                title="Deepfake Video Detected",
                description=f"Detected potential deepfake video with {prediction[1]}% confidence.",
                source=videoPath,
                confidence=prediction[1]/100,
                evidence=json.dumps({
                    'prediction': prediction,
                    'processing_time': processing_time,
                    'video_path': videoPath
                }, indent=2)
            )
        
        return prediction, processing_time
    except Exception as e:
        logger.error(f"Error in detectFakeVideo: {str(e)}")
        traceback.print_exc()
        raise

def get_datasets():
    datasets = []
    for item in os.listdir(DATASET_FOLDER):
        if item.endswith('.zip'):
            path = os.path.join(DATASET_FOLDER, item)
            stats = os.stat(path)
            datasets.append({
                'name': item,
                'size': stats.st_size,
                'upload_date': datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    return datasets

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'GET':
        return render_template('detect.html')
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('detect.html', error="No video file uploaded")

        video = request.files['video']
        if video.filename == '':
            return render_template('detect.html', error="No video file selected")

        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return render_template('detect.html',
                                   error="Invalid file format. Please upload MP4, AVI, or MOV files.")

        video_filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)

        try:
            logger.info(f"Processing video: {video_filename}")

            frames, frame_paths = extract_frames(video_path)

            if not frames:
                raise Exception("No frames could be extracted from the video")

            prediction, processing_time = detectFakeVideo(video_path)

            if prediction[0] == 0:
                output = "FAKE"
            else:
                output = "REAL"
            confidence = prediction[1]

            logger.info(f"Video prediction: {output} with confidence {confidence}%")

            confidence_image = generate_confidence_graph(confidence)
            if not confidence_image:
                raise Exception("Failed to generate confidence graph")

            comparison_image = generate_comparison_graph(confidence)
            if not comparison_image:
                raise Exception("Failed to generate comparison graph")

            data = {
                'output': output,
                'confidence': confidence,
                'frames': frame_paths,
                'processing_time': round(processing_time, 2),
                'confidence_image': confidence_image,
                'comparison_image': comparison_image
            }

            logger.info(f"Sending response data: {data}")
            data = json.dumps(data)

            os.remove(video_path)
            return render_template('detect.html', data=data)

        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            error_msg = str(e)
            logger.error(f"Error processing video: {error_msg}")
            traceback.print_exc()
            return render_template('detect.html', error=f"Error processing video: {error_msg}")

@app.route('/detect_phishing_route', methods=['GET', 'POST'])
@login_required
def detect_phishing_route():
    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        content_type = request.form.get('content_type', 'url')

        if not content:
            return render_template('detect_phishing.html', error="Please provide content to analyze")

        try:
            # Validate content type
            if content_type not in ['url', 'email']:
                return render_template('detect_phishing.html', error="Invalid content type")

            # Validate URL format if content type is URL
            if content_type == 'url':
                # Add http:// if no scheme is provided
                if not content.startswith(('http://', 'https://')):
                    content = 'http://' + content
                
                try:
                    parsed = urlparse(content)
                    if not parsed.netloc:
                        return render_template('detect_phishing.html', error="Invalid URL format")
                    
                    # Reconstruct the URL to ensure it's properly formatted
                    content = urlunparse(parsed)
                except Exception as e:
                    logger.error(f"URL parsing error: {str(e)}")
                    return render_template('detect_phishing.html', error="Invalid URL format")

            # Perform phishing detection
            results = detect_phishing(content, content_type)
            return render_template('detect_phishing.html', results=results)

        except Exception as e:
            logger.error(f"Error in phishing detection: {str(e)}")
            return render_template('detect_phishing.html', error=f"Error analyzing content: {str(e)}")

    return render_template('detect_phishing.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get all alerts for the current user
    alerts = SecurityAlert.query.filter_by(user_id=current_user.id).order_by(SecurityAlert.timestamp.desc()).all()
    
    # Calculate statistics
    stats = {
        'total_alerts': len(alerts),
        'critical_alerts': len([a for a in alerts if a.severity == 'critical']),
        'pending_alerts': len([a for a in alerts if a.status in ['new', 'in_progress']])
    }
    
    return render_template('dashboard.html', alerts=alerts, stats=stats)

@app.route('/api/alerts/<int:alert_id>/status', methods=['PUT'])
@login_required
def update_alert_status(alert_id):
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        new_status = data.get('status')
        
        if not new_status:
            return jsonify({'success': False, 'error': 'Status is required'}), 400
        
        if new_status not in ['new', 'in_progress', 'resolved', 'dismissed']:
            return jsonify({'success': False, 'error': 'Invalid status value'}), 400
        
        # Get alert and validate ownership
        alert = SecurityAlert.query.get_or_404(alert_id)
        if alert.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'Unauthorized access'}), 403
        
        # Validate status transition
        valid_transitions = {
            'new': ['in_progress', 'resolved', 'dismissed'],
            'in_progress': ['resolved', 'dismissed'],
            'resolved': ['in_progress'],
            'dismissed': ['in_progress']
        }
        
        if new_status not in valid_transitions.get(alert.status, []):
            return jsonify({
                'success': False, 
                'error': f'Cannot transition from {alert.status} to {new_status}'
            }), 400
        
        # Update alert status
        alert.status = new_status
        if new_status == 'resolved':
            alert.resolved_at = datetime.utcnow()
        
        try:
            db.session.commit()
            logger.info(f"Alert {alert_id} status updated to {new_status} by user {current_user.id}")
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database error updating alert status: {str(e)}")
            return jsonify({'success': False, 'error': 'Database error occurred'}), 500
            
    except Exception as e:
        logger.error(f"Error updating alert status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/investigate/<int:alert_id>')
@login_required
def investigate_alert(alert_id):
    alert = SecurityAlert.query.get_or_404(alert_id)
    
    # Check if the alert belongs to the current user
    if alert.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))
    
    # Update alert status to in_progress if it's new
    if alert.status == 'new':
        alert.status = 'in_progress'
        db.session.commit()
    
    # Redirect to appropriate investigation page based on alert type
    if alert.alert_type == 'phishing':
        return redirect(url_for('detect_phishing_route'))
    elif alert.alert_type == 'deepfake':
        return redirect(url_for('detect'))
    else:
        # For other types, show a generic investigation page
        return render_template('investigate.html', alert=alert)

def create_security_alert(user_id, alert_type, severity, title, description, source=None, confidence=None, evidence=None):
    """
    Helper function to create a new security alert
    """
    try:
        alert = SecurityAlert(
            user_id=user_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            source=source,
            confidence=confidence,
            evidence=evidence,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string,
            status='new'
        )
        db.session.add(alert)
        db.session.commit()
        return alert
    except Exception as e:
        logger.error(f"Error creating security alert: {str(e)}")
        return None

# Modify the detect_phishing function to create alerts
def detect_phishing(content, content_type):
    results = {
        'is_phishing': False,
        'risk_level': 'Low',
        'warnings': [],
        'details': {},
        'confidence': 0.0
    }
    
    try:
        # Test mode - Check if content contains test markers
        if content.strip().startswith('TEST:'):
            test_parts = content.strip().split(':', 1)
            if len(test_parts) == 2:
                test_type = test_parts[1].strip().lower()
                if test_type == 'phishing':
                    results['is_phishing'] = True
                    results['risk_level'] = 'High'
                    results['confidence'] = 0.70  # 70% confidence for phishing
                    results['warnings'].append('Test mode: Simulated phishing detection')
                    return results
                elif test_type == 'real':
                    results['is_phishing'] = False
                    results['risk_level'] = 'Low'
                    results['confidence'] = 0.99  # 99% confidence for real
                    results['warnings'].append('Test mode: Simulated real content')
                    return results

        # List of known legitimate domains
        legitimate_domains = {
            'google.com': 0.99,
            'facebook.com': 0.99,
            'amazon.com': 0.99,
            'apple.com': 0.99,
            'microsoft.com': 0.99,
            'paypal.com': 0.99,
            'netflix.com': 0.99,
            'twitter.com': 0.99,
            'linkedin.com': 0.99,
            'github.com': 0.99,
            'yahoo.com': 0.99,
            'outlook.com': 0.99,
            'gmail.com': 0.99,
            'hotmail.com': 0.99,
            'instagram.com': 0.99,
            'youtube.com': 0.99,
            'wikipedia.org': 0.99,
            'reddit.com': 0.99,
            'spotify.com': 0.99,
            'dropbox.com': 0.99
        }

        warning_weights = {
            'suspicious_tld': 0.3,
            'ip_address': 0.4,
            'unusual_subdomains': 0.2,
            'phishing_keywords': 0.15,
            'no_https': 0.2,
            'urgency_keywords': 0.25,
            'embedded_urls': 0.2,
            'mismatched_names': 0.3,
            'suspicious_characters': 0.15,
            'short_domain': 0.1,
            'typo_squatting': 0.25,
            'random_domain': 0.4
        }
        
        total_weight = 0.0
        detected_issues = 0
        content_length = len(content)
        
        if content_type == 'url':
            # URL-specific checks
            parsed = urlparse(content)
            domain = parsed.netloc.lower()
            
            # Check if it's a known legitimate domain
            if domain in legitimate_domains:
                results['is_phishing'] = False
                results['risk_level'] = 'Low'
                results['confidence'] = legitimate_domains[domain]
                results['warnings'].append('Domain is a known legitimate website')
                return results
            
            # Check for random/unknown domain
            if len(domain) > 10 and not any(known in domain for known in legitimate_domains.keys()):
                results['warnings'].append('Unknown domain name')
                results['risk_level'] = 'High'
                total_weight += warning_weights['random_domain']
                detected_issues += 1
            
            # Check for suspicious TLDs
            suspicious_tlds = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq']
            if any(parsed.netloc.endswith(tld) for tld in suspicious_tlds):
                results['warnings'].append('Suspicious TLD detected')
                results['risk_level'] = 'High'
                total_weight += warning_weights['suspicious_tld']
                detected_issues += 1
            
            # Check for IP address instead of domain
            ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
            if re.match(ip_pattern, parsed.netloc):
                results['warnings'].append('URL contains IP address instead of domain name')
                results['risk_level'] = 'High'
                total_weight += warning_weights['ip_address']
                detected_issues += 1
            
            # Check for unusual subdomains
            if parsed.netloc.count('.') > 2:
                results['warnings'].append('Unusual number of subdomains')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['unusual_subdomains']
                detected_issues += 1
            
            # Check for common phishing keywords in path
            phishing_keywords = ['login', 'signin', 'account', 'verify', 'secure', 'bank', 'paypal']
            if any(keyword in parsed.path.lower() for keyword in phishing_keywords):
                results['warnings'].append('Contains common phishing keywords in URL path')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['phishing_keywords']
                detected_issues += 1
            
            # Check for HTTPS
            if parsed.scheme != 'https':
                results['warnings'].append('Not using HTTPS')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['no_https']
                detected_issues += 1

            # Check for suspicious characters in domain
            suspicious_chars = re.findall(r'[^a-zA-Z0-9.-]', parsed.netloc)
            if suspicious_chars:
                results['warnings'].append('Domain contains suspicious characters')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['suspicious_characters'] * len(suspicious_chars)
                detected_issues += 1

            # Check for very short domain names
            if len(parsed.netloc) < 8:
                results['warnings'].append('Suspiciously short domain name')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['short_domain']
                detected_issues += 1

            # Check for typo squatting (common domain names with slight variations)
            domain_base = parsed.netloc.split('.')[0]
            for common in legitimate_domains.keys():
                common_base = common.split('.')[0]
                if domain_base != common_base and len(domain_base) > 3:
                    # Calculate Levenshtein distance
                    if sum(a != b for a, b in zip(domain_base, common_base)) <= 2:
                        results['warnings'].append(f'Possible typo squatting of {common}')
                        results['risk_level'] = 'High'
                        total_weight += warning_weights['typo_squatting']
                        detected_issues += 1
                        break
            
            results['details']['domain'] = parsed.netloc
            results['details']['path'] = parsed.path
            
        elif content_type == 'email':
            # Email-specific checks
            # Check for urgency keywords
            urgency_keywords = ['urgent', 'immediate', 'action required', 'account suspended', 
                              'verify now', 'click here', 'confirm now']
            urgency_count = sum(1 for keyword in urgency_keywords if keyword in content.lower())
            if urgency_count > 0:
                results['warnings'].append(f'Contains {urgency_count} urgency-inducing phrases')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['urgency_keywords'] * min(urgency_count, 3)
                detected_issues += 1
            
            # Check for suspicious links
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, content)
            if urls:
                results['details']['found_urls'] = urls
                results['warnings'].append(f'Contains {len(urls)} embedded URLs')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['embedded_urls'] * min(len(urls), 3)
                detected_issues += 1
            
            # Check for mismatched sender/display names
            if '>' in content and '<' in content:
                results['warnings'].append('Potential mismatched sender/display names')
                results['risk_level'] = 'Medium'
                total_weight += warning_weights['mismatched_names']
                detected_issues += 1

            # Check for suspicious patterns in email content
            suspicious_patterns = [
                (r'\b(?:password|account|verify|confirm|login)\b', 0.15),
                (r'\b(?:click here|download now|free offer)\b', 0.2),
                (r'\b(?:congratulations|winner|prize|lottery)\b', 0.25)
            ]
            
            for pattern, weight in suspicious_patterns:
                matches = re.findall(pattern, content.lower())
                if matches:
                    results['warnings'].append(f'Contains suspicious phrases: {", ".join(set(matches))}')
                    total_weight += weight * min(len(matches), 3)
                    detected_issues += 1
        
        # Calculate confidence score based on multiple factors
        if detected_issues > 0:
            # Base confidence on the total weight of detected issues
            base_confidence = min(1.0, total_weight)
            
            # Adjust confidence based on number of issues
            issue_multiplier = min(1.0, detected_issues * 0.15)  # Each issue adds 15% up to 100%
            
            # Adjust confidence based on risk level
            risk_multiplier = {
                'Low': 0.3,
                'Medium': 0.6,
                'High': 0.9
            }[results['risk_level']]
            
            # Calculate final confidence
            results['confidence'] = min(1.0, (base_confidence + issue_multiplier) * risk_multiplier)
        else:
            # If no issues detected, confidence is very low
            results['confidence'] = 0.1
        
        # Determine if it's phishing based on risk level and warnings
        if results['risk_level'] in ['High', 'Medium'] and len(results['warnings']) > 0:
            results['is_phishing'] = True
        
        # Create security alert if phishing is detected
        if results['is_phishing'] and current_user.is_authenticated:
            severity = 'critical' if results['risk_level'] == 'High' else 'high'
            create_security_alert(
                user_id=current_user.id,
                alert_type='phishing',
                severity=severity,
                title=f"Phishing Attempt Detected - {content_type.upper()}",
                description=f"Detected potential phishing attempt in {content_type} with {len(results['warnings'])} suspicious indicators.",
                source=content,
                confidence=results['confidence'],
                evidence=json.dumps(results, indent=2)
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in phishing detection: {str(e)}")
        raise

if __name__ == '__main__':
    app.run(port=3000, debug=True)

