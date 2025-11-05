from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2
import base64
import datetime
import json
import re
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import random
import math

app = Flask(__name__)
app.secret_key = 'alzheimers_detection_secret_key'

# Database simulation (in a real app, use a proper database)
users_db = {}
items_db = {}
heatmap_data = {}  # Store heatmap data for each user

# Load the Alzheimer's detection model
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

# Try to load the model
try:
    model = load_model()
    print("Model loaded successfully")
    # Use the same image size as Streamlit app
    IMG_SIZE = (128, 128)
    print(f"Model input shape: {IMG_SIZE}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    IMG_SIZE = (128, 128)  # Use the same size as Streamlit app

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def preprocess_image(image):
    """Match the exact preprocessing from Streamlit app"""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Convert grayscale to RGB if needed (same as Streamlit logic)
    if len(img_array.shape) == 2:  # Grayscale image
        rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
    elif img_array.shape[2] == 1:  # Single channel
        rgb_image = np.repeat(img_array, 3, axis=2)
    else:  # Already RGB
        rgb_image = img_array
    
    # Resize to match model input size
    img_pil = Image.fromarray(rgb_image.astype('uint8'))
    img_pil = img_pil.resize(IMG_SIZE)
    
    # Convert back to array and expand dimensions for model
    img_array = np.array(img_pil)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if username in users_db:
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        # Store user data (in a real app, store in a database)
        users_db[username] = {
            'password': generate_password_hash(password),
            'email': email
        }
        
        # Initialize heatmap data for the user
        heatmap_data[username] = []
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users_db and check_password_hash(users_db[username]['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/detection', methods=['GET', 'POST'])
@login_required
def detection():
    prediction = None
    confidence = None
    
    if request.method == 'POST' and 'scan' in request.files:
        scan = request.files['scan']
        if scan.filename != '':
            # Process the image using the same logic as Streamlit
            img = Image.open(scan.stream)
            
            # Preprocess the image (same as Streamlit)
            img_array = preprocess_image(img)
            
            # Make prediction
            try:
                predictions = model.predict(img_array)
                predicted_idx = np.argmax(predictions, axis=1)[0]
                confidence = float(predictions[0][predicted_idx]) * 100
                
                # Use the same class labels as Streamlit
                class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
                prediction = class_labels[predicted_idx]
                
                # Store the prediction in session
                session['prediction'] = prediction
                session['confidence'] = confidence
                
                return redirect(url_for('results'))
            except Exception as e:
                flash(f'Error making prediction: {str(e)}', 'danger')
    
    return render_template('detection.html')

@app.route('/results')
@login_required
def results():
    prediction = session.get('prediction', None)
    confidence = session.get('confidence', None)
    
    if not prediction:
        flash('No prediction available. Please upload a scan first.', 'warning')
        return redirect(url_for('detection'))
    
    # Map the prediction to match the recommendation keys
    prediction_map = {
        'Mild_Demented': 'MildDemented',
        'Moderate_Demented': 'ModerateDemented', 
        'Non_Demented': 'NonDemented',
        'Very_Mild_Demented': 'VeryMildDemented'
    }
    
    mapped_prediction = prediction_map.get(prediction, prediction)
    
    # Recommendations based on prediction
    recommendations = get_recommendations(mapped_prediction)
    
    return render_template('results.html', 
                          prediction=prediction, 
                          confidence=confidence,
                          recommendations=recommendations)

def get_recommendations(prediction):
    recommendations = {
        'MildDemented': {
            'description': 'Early stage Alzheimer\'s disease with mild cognitive impairment.',
            'doctors': ['Neurologist', 'Geriatrician'],
            'medications': ['Cholinesterase inhibitors (Aricept, Exelon)'],
            'lifestyle': [
                'Regular cognitive exercises',
                'Establish daily routines',
                'Memory aids like calendars and to-do lists',
                'Regular physical exercise'
            ]
        },
        'ModerateDemented': {
            'description': 'Moderate stage Alzheimer\'s with significant memory and cognitive issues.',
            'doctors': ['Neurologist', 'Geriatrician', 'Psychiatrist'],
            'medications': [
                'Cholinesterase inhibitors (Aricept, Exelon)',
                'Memantine (Namenda)',
                'Combination therapies'
            ],
            'lifestyle': [
                'Supervised daily activities',
                'Simplified communication',
                'Safety-proofing the home',
                'Caregiver support and education'
            ]
        },
        'NonDemented': {
            'description': 'No signs of Alzheimer\'s disease detected.',
            'doctors': ['Primary care physician for regular check-ups'],
            'medications': ['No specific Alzheimer\'s medications needed'],
            'lifestyle': [
                'Regular physical exercise',
                'Healthy diet rich in antioxidants',
                'Mental stimulation and cognitive activities',
                'Social engagement',
                'Regular health check-ups'
            ]
        },
        'VeryMildDemented': {
            'description': 'Very early stage Alzheimer\'s with subtle cognitive changes.',
            'doctors': ['Neurologist', 'Geriatrician'],
            'medications': ['Cholinesterase inhibitors may be considered'],
            'lifestyle': [
                'Cognitive training exercises',
                'Memory aids and organization tools',
                'Regular physical exercise',
                'Heart-healthy diet',
                'Social activities'
            ]
        }
    }
    
    return recommendations.get(prediction, {})

# ... rest of your existing code remains exactly the same ...

@app.route('/item_tracker')
@login_required
def item_tracker():
    username = session['username']
    user_items = items_db.get(username, [])
    
    # Get user's heatmap data
    user_heatmap = heatmap_data.get(username, [])
    
    # Get room zones for the heatmap visualization
    room_zones = get_room_zones()
    
    return render_template('item_tracker.html', 
                          items=user_items, 
                          heatmap_data=json.dumps(user_heatmap),
                          room_zones=json.dumps(room_zones))

def get_room_zones():
    """Define room zones for the heatmap visualization"""
    zones = [
        {"name": "Living Room", "x": 50, "y": 150, "width": 300, "height": 200},
        {"name": "Kitchen", "x": 400, "y": 150, "width": 250, "height": 200},
        {"name": "Bedroom", "x": 50, "y": 400, "width": 250, "height": 180},
        {"name": "Bathroom", "x": 350, "y": 400, "width": 150, "height": 180},
        {"name": "Dining Area", "x": 550, "y": 400, "width": 200, "height": 180},
        {"name": "Entryway", "x": 700, "y": 150, "width": 150, "height": 200}
    ]
    return zones

@app.route('/capture_item', methods=['POST'])
@login_required
def capture_item():
    if request.method == 'POST':
        data = request.get_json()
        username = session['username']
        
        if username not in items_db:
            items_db[username] = []
        
        # Process the image data
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        # Convert to image for processing
        img = Image.open(io.BytesIO(img_bytes))
        
        # Generate a new item ID
        new_id = len(items_db[username]) + 1
        
        # Get location coordinates for heatmap
        location_x = data.get('locationX', random.randint(50, 750))
        location_y = data.get('locationY', random.randint(150, 550))
        
        # Update heatmap data
        if username not in heatmap_data:
            heatmap_data[username] = []
        
        heatmap_data[username].append({
            "x": location_x,
            "y": location_y,
            "value": 1,
            "radius": 30,
            "itemName": data['itemName'],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Save item information
        item_info = {
            'id': new_id,
            'name': data['itemName'],
            'location': data['location'],
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image': data['image'],  # Store base64 image
            'hand_detections': [],  # Initialize empty hand detections array
            'location_x': location_x,
            'location_y': location_y,
            'room_zone': determine_room_zone(location_x, location_y)
        }
        
        items_db[username].append(item_info)
        
        return jsonify({'success': True, 'message': 'Item captured successfully', 'itemId': new_id})

def determine_room_zone(x, y):
    """Determine which room zone the coordinates fall into"""
    zones = get_room_zones()
    
    for zone in zones:
        if (x >= zone['x'] and x <= zone['x'] + zone['width'] and 
            y >= zone['y'] and y <= zone['y'] + zone['height']):
            return zone['name']
    
    return "Unknown Area"

@app.route('/hand_detection_data', methods=['POST'])
@login_required
def hand_detection_data():
    if request.method == 'POST':
        data = request.get_json()
        username = session['username']
        
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store hand detection data with the item
        if 'itemId' in data and username in items_db:
            for item in items_db[username]:
                if item['id'] == data['itemId']:
                    if 'hand_detections' not in item:
                        item['hand_detections'] = []
                    
                    hand_data = {
                        'timestamp': timestamp,
                        'hand_position': data['handPosition'],
                        'confidence': data['confidence'],
                        'gesture': data.get('gesture', 'Unknown')
                    }
                    
                    item['hand_detections'].append(hand_data)
                    return jsonify({'success': True, 'message': 'Hand detection data saved'})
        
        return jsonify({'success': False, 'message': 'Item not found'})

@app.route('/delete_item/<int:item_id>', methods=['POST'])
@login_required
def delete_item(item_id):
    username = session['username']
    
    if username in items_db:
        # Find the item to delete
        item_to_delete = None
        for item in items_db[username]:
            if item['id'] == item_id:
                item_to_delete = item
                break
        
        # Remove the item
        if item_to_delete:
            items_db[username] = [item for item in items_db[username] if item['id'] != item_id]
            
            # Also remove from heatmap data
            if username in heatmap_data:
                # Find and remove the corresponding heatmap point
                for i, point in enumerate(heatmap_data[username]):
                    if point.get('itemName') == item_to_delete.get('name') and point.get('timestamp') == item_to_delete.get('timestamp'):
                        heatmap_data[username].pop(i)
                        break
        
    return redirect(url_for('item_tracker'))

@app.route('/toggle_auto_tracking', methods=['POST'])
@login_required
def toggle_auto_tracking():
    username = session['username']
    auto_tracking_enabled = request.json.get('enabled', False)
    
    # Store the auto-tracking preference in the session
    session['auto_tracking_enabled'] = auto_tracking_enabled
    
    return jsonify({
        'success': True, 
        'auto_tracking_enabled': auto_tracking_enabled,
        'message': 'Auto tracking ' + ('enabled' if auto_tracking_enabled else 'disabled')
    })

@app.route('/auto_capture_item', methods=['POST'])
@login_required
def auto_capture_item():
    if request.method == 'POST':
        data = request.get_json()
        username = session['username']
        
        if username not in items_db:
            items_db[username] = []
        
        # Process the image data
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        # Convert to image for processing
        img = Image.open(io.BytesIO(img_bytes))
        
        # Generate a new item ID
        new_id = len(items_db[username]) + 1
        
        # Auto-generate item name based on time if not provided
        item_name = data.get('itemName', f"Auto-detected item {new_id}")
        
        # Auto-generate location based on data if not provided
        location = data.get('location', "Location not specified")
        
        # Get location coordinates for heatmap
        location_x = data.get('locationX', random.randint(50, 750))
        location_y = data.get('locationY', random.randint(150, 550))
        
        # Update heatmap data
        if username not in heatmap_data:
            heatmap_data[username] = []
        
        heatmap_data[username].append({
            "x": location_x,
            "y": location_y,
            "value": 1,
            "radius": 30,
            "itemName": item_name,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Determine room zone
        room_zone = determine_room_zone(location_x, location_y)
        
        # Save item information
        item_info = {
            'id': new_id,
            'name': item_name,
            'location': location,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image': data['image'],  # Store base64 image
            'hand_detections': [],  # Initialize empty hand detections array
            'auto_captured': True,  # Flag to indicate this was auto-captured
            'location_x': location_x,
            'location_y': location_y,
            'room_zone': room_zone,
            'gesture': data.get('gesture', 'Unknown')
        }
        
        items_db[username].append(item_info)
        
        return jsonify({
            'success': True, 
            'message': 'Item auto-captured successfully', 
            'itemId': new_id,
            'roomZone': room_zone
        })

@app.route('/get_heatmap_data')
@login_required
def get_heatmap_data():
    username = session['username']
    user_heatmap = heatmap_data.get(username, [])
    
    # Aggregate data to show frequency
    aggregated_data = {}
    for point in user_heatmap:
        key = f"{point['x']},{point['y']}"
        if key in aggregated_data:
            aggregated_data[key]['value'] += 1
        else:
            aggregated_data[key] = {
                'x': point['x'],
                'y': point['y'],
                'value': 1,
                'radius': 30
            }
    
    return jsonify(list(aggregated_data.values()))

@app.route('/get_item_statistics')
@login_required
def get_item_statistics():
    username = session['username']
    user_items = items_db.get(username, [])
    
    # Calculate statistics
    total_items = len(user_items)
    auto_captured = sum(1 for item in user_items if item.get('auto_captured', False))
    
    # Count items by room zone
    room_counts = {}
    for item in user_items:
        zone = item.get('room_zone', 'Unknown Area')
        if zone in room_counts:
            room_counts[zone] += 1
        else:
            room_counts[zone] = 1
    
    # Most common items
    item_counts = {}
    for item in user_items:
        name = item.get('name', 'Unknown')
        if name in item_counts:
            item_counts[name] += 1
        else:
            item_counts[name] = 1
    
    most_common_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Time-based statistics
    morning_items = sum(1 for item in user_items if '06:00:00' <= item.get('timestamp', '')[-8:] <= '11:59:59')
    afternoon_items = sum(1 for item in user_items if '12:00:00' <= item.get('timestamp', '')[-8:] <= '17:59:59')
    evening_items = sum(1 for item in user_items if '18:00:00' <= item.get('timestamp', '')[-8:] <= '23:59:59')
    night_items = sum(1 for item in user_items if '00:00:00' <= item.get('timestamp', '')[-8:] <= '05:59:59')
    
    return jsonify({
        'total_items': total_items,
        'auto_captured': auto_captured,
        'manual_captured': total_items - auto_captured,
        'room_counts': room_counts,
        'most_common_items': most_common_items,
        'time_stats': {
            'morning': morning_items,
            'afternoon': afternoon_items,
            'evening': evening_items,
            'night': night_items
        }
    })

if __name__ == '__main__':
    app.run(debug=True)