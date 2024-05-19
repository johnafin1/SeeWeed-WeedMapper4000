import os
import secrets
import shutil
from glob import glob
from utils.flight import Flight
import logging
import mimetypes
import pandas as pd
from dotenv import load_dotenv

from flask import Flask, jsonify, send_from_directory, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from utils.utils import (allowed_file, allowed_movie_file, seed_everything,
                         check_hardware, get_weight_path, get_pred_folder_path,
                         get_splitter, get_user_csv_paths, allowed_csv_file, get_user_uploads_paths, get_map_data_folder_path,
                         get_prediction_data_folder_path, get_flight_folder_path, ALLOWED_MOVIE_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS, ALLOWED_EXTENSIONS)
from predict import Yolov8
import json

load_dotenv()
# Seed for reproducibility
seed_everything()
logging.basicConfig(level=logging.INFO)
# Flask application setup
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key

def basename_filter(s):
    return os.path.basename(s)

# Add the filter to Jinja environment
app.jinja_env.filters['basename'] = basename_filter
# Path configurations
upload_img_folder = './uploads/user-uploads/'
results_path = './uploads/results/'
demo_results_path = './uploads/demo_results/'

# Glob patterns to collect file paths
pred_test_img_paths_jpeg = glob('./uploads/test-set/images/*.jpeg')
pred_test_img_paths_jpg = glob('./uploads/test-set/images/*.jpg')
pred_test_img_paths = pred_test_img_paths_jpeg + pred_test_img_paths_jpg

pred_img_paths_jpeg = glob('./uploads/user-uploads/*.jpeg')
pred_img_paths_jpg = glob('./uploads/user-uploads/*.jpg')
pred_img_paths_mp4 = glob('./uploads/user-uploads/*.mp4')
pred_img_paths_mov = glob('./uploads/user-uploads/*.mov')
pred_img_paths = pred_img_paths_jpeg + pred_img_paths_jpg + pred_img_paths_mp4 + pred_img_paths_mov

# Model initialization
model = Yolov8(pred_img_paths, results_path)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def get_images_data():
    images_with_pred_data = []
    images_data = []

    # Directory scanning and file handling
    for filename in os.listdir(get_user_uploads_paths()):
        if filename.split('.')[-1].lower() in ALLOWED_IMAGE_EXTENSIONS:
            original_image_path = os.path.join(get_user_uploads_paths(), filename)
            processed_image_path = os.path.join(get_pred_folder_path(), filename)
            stats_file_path = os.path.join(get_prediction_data_folder_path(), filename + '.json')  # Assuming JSON files are stored here

            if os.path.exists(processed_image_path):
                # Try to read prediction stats from the JSON file
                if os.path.exists(stats_file_path):
                    with open(stats_file_path, 'r') as file:
                        prediction_stats = json.load(file)
                    detection_time = prediction_stats[-1]['prediction_time']  # Get the most recent entry
                    detection_count = prediction_stats[-1]['detection_count']
                else:
                    detection_time = "N/A"
                    detection_count = "N/A"

                images_with_pred_data.append((original_image_path, processed_image_path, detection_time, detection_count))
            else:
                images_data.append((original_image_path))
    return images_with_pred_data, images_data


def create_or_update_flight():
    data = request.form
    try:
        # Assume data includes 'name', 'video_path', and optionally 'telemetry_path'
        flight_name = data.get('name')
        video_path = data.get('video_path')
        telemetry_path = data.get('telemetry_path', None)

        # Attempt to create or update a flight
        flight = Flight.create_new(flight_name, video_path, telemetry_path=telemetry_path)
        if 'run_detection' in data:
            flight.run_detection()
        if telemetry_path:
            flight.add_telemetry(telemetry_path)

        flight.save_flight()
        return jsonify({'success': True, 'message': 'Flight processed successfully.'}), 200
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def upload_and_process_file():
    if 'file[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('file[]')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_img_folder, filename)
            try:
                file.save(file_path)
                # Here you would trigger any specific model processing
                flash('File(s) successfully uploaded')
            except Exception as e:
                flash(f"Error saving file: {e}")
                continue
        else: 
            flash('File type not allowed')
            return redirect(request.url)

    # Here you can redirect to a new page that handles further processing or displaying
    return redirect(url_for('display_uploads'))

@app.route('/upload_flight', methods=['POST'])
def upload_flight():
    if 'videoFile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    video_file = request.files['videoFile']
    flight_name = request.form['flightName']
    logging.info(f"Flight details saved for {flight_name} and {video_file}")

    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(get_user_uploads_paths(), filename)
        video_file.save(video_path)
        logging.info(f"Flight details saved for {filename}")

        try:
            # Create a new Flight instance and save it
            new_flight = Flight.create_new(flight_name, video_path)
            new_flight.save_flight()
            flash('Flight successfully uploaded and created')
        except Exception as e:
            flash('Error creating new flight')
        return redirect(url_for('display_flights'))
    else:
        flash('Invalid file type')
        return redirect(request.url)

@app.route('/flights', methods=['GET'])
def display_flights():
    try:
        flights_info = []
        flights_folder = get_flight_folder_path()
        for filename in os.listdir(flights_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(flights_folder, filename)
                with open(file_path, 'r') as file:
                    flight_data = json.load(file)
                    # Check if telemetry data has been uploaded by verifying the telemetry_path
                    telemetry_path = flight_data.get('telemetry_path', None)
                    flight_data['telemetry_uploaded'] = os.path.exists(telemetry_path) if telemetry_path else False
                    flights_info.append(flight_data)

        return render_template('flights.html', flights=flights_info)
    except Exception as e:
        logging.error(f"Failed to retrieve flights: {str(e)}")
        return jsonify({'success': False, 'message': 'Failed to retrieve flights'}), 500
    
@app.route('/run_prediction/<flight_name>', methods=['POST'])
def run_prediction(flight_name):
    # Locate the correct flight instance or reload it
    flight_path = os.path.join(get_flight_folder_path(), f"{flight_name}.json")
    if os.path.exists(flight_path):
        with open(flight_path, 'r') as file:
            flight_data = json.load(file)
        flight = Flight.load_from_json(flight_data)
        flight.run_detection()
        flight.save_flight()
        return jsonify({'success': True, 'message': 'Prediction started successfully.'})
    return jsonify({'success': False, 'message': 'Flight not found.'}), 404

@app.route('/upload_telemetry/<flight_name>', methods=['POST'])
def upload_telemetry(flight_name):
    if 'telemetryFile' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['telemetryFile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_csv_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(get_user_csv_paths(), filename)
        file.save(file_path)

        # Convert Excel to CSV if necessary
        if filename.endswith(('.xls', '.xlsx')):
            # Read the Excel file
            df = pd.read_excel(file_path)
            # Convert to CSV
            new_filename = filename.rsplit('.', 1)[0] + '.csv'
            new_file_path = os.path.join(get_user_csv_paths(), new_filename)
            df.to_csv(new_file_path, index=False)
            # Update file_path to point to the new CSV
            file_path = new_file_path

        # Load and update the flight instance
        flight = Flight.get_flight_by_name(flight_name)  # Assuming there's a method to load a Flight instance by name
        flight.add_telemetry(file_path)
        flight.save_flight()  # Ensure there's a method to save the updated Flight data

        flash('Telemetry data uploaded successfully.')
        return redirect(url_for('display_flights'))

    flash('Invalid file type.')
    return redirect(request.url)

@app.route('/generate_map/<flight_name>', methods=['POST'])
def generate_map(flight_name):
    logging.info("Started map gen")
    flight = Flight.get_flight_by_name(flight_name)
    logging.info("Got Flight map gen {flight_name}")
    if flight:
        try:
            flight.generate_map()
            map_path = flight.get_map_data_path()
            map_html = flight.generate_map_html()
            if map_path:
                return jsonify({'success': True, 'message': 'Map generated successfully.', 'map_url': url_for('display_map', map_filename=map_path)})
            else:
                return jsonify({'success': False, 'message': 'Map generation failed.'})
        except Exception as e:
            logging.error(f"Failed to generate map for {flight_name}: {str(e)}")
            return jsonify({'success': False, 'message': 'Map generation failed due to an error.'})
    else:
        return jsonify({'success': False, 'message': 'Flight not found.'})

@app.route('/uploads', methods=['GET'])
def display_uploads():
    images_with_pred_data, images_data = get_images_data()
    return render_template('uploads.html', images_with_pred_data=images_with_pred_data, images_data=images_data)

@app.route('/display/heat_map/<flight_name>')
def display_map(flight_name):
    # Assuming you load map data here, which contains latitude, longitude, and detection counts
    try:
        flight = Flight.get_flight_by_name(flight_name)
        if flight:
            # Load the map data JSON
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                logging.info("API Key loaded successfully")
            else:
                logging.info("API Key not found")
            map_data = flight.load_map_data()  # This method needs to be implemented to read the JSON data file
            return render_template('objectHeatGoogleMap.html', map_data=map_data, flight_name=flight.name, api_key = api_key)
        else:
            flash('Flight not found.')
            return redirect(url_for('flights'))
    except Exception as e:
        logging.error(f"Error displaying map for {flight_name}: {str(e)}")
        flash('Error occurred while trying to display the map.')
        return redirect(url_for('flights'))
    

@app.route('/display/flight_upload/<filename>')
def display_upload_video(filename):
    safe_filename = secure_filename(filename)
    upload_folder = os.path.join(app.root_path, 'uploads', 'user-uploads')

    # Customize MIME type based on file extension
    extension = safe_filename.split('.')[-1].lower()  # Get extension and normalize it to lowercase
    if extension == "mp4":
        mime_type = 'video/mp4'
    elif extension == "mov":
        mime_type = 'video/quicktime'
    elif extension == "avi":
        mime_type = 'video/x-msvideo'
    else:
        mime_type = 'application/octet-stream'  # Fallback MIME type

    try:
        return send_from_directory(upload_folder, safe_filename, mimetype=mime_type)
    except Exception as e:
        return str(e), 404

@app.route('/display/flight_pred/<filename>')
def display_pred_video(filename):
    safe_filename = secure_filename(filename)
    processed_filename = f"{os.path.splitext(safe_filename)[0]}_processed{os.path.splitext(safe_filename)[1]}"
    result_folder = os.path.join(app.root_path, 'uploads', 'results')
    # Customize MIME type based on file extension
    extension = processed_filename.split('.')[-1].lower()  # Get extension and normalize it to lowercase
    if extension == "mp4":
        mime_type = 'video/mp4'
    elif extension == "mov":
        mime_type = 'video/quicktime'
    elif extension == "avi":
        mime_type = 'video/x-msvideo'
    else:
        mime_type = 'application/octet-stream'  # Fallback MIME type

    try:
        # First try to serve the processed video if it exists
        if os.path.exists(os.path.join(result_folder, processed_filename)):
            return send_from_directory(result_folder, processed_filename, mime_type=mime_type)
        # Fall back to the original video if the processed one isn't available
        elif os.path.exists(os.path.join(result_folder, safe_filename)):
            return send_from_directory(result_folder, safe_filename, mime_type=mime_type)
        else:
            raise FileNotFoundError("Video not found")
    except Exception as e:
        return str(e), 404


@app.route('/display/upload/<filename>')
def display_upload_image(filename):
    safe_filename = secure_filename(filename)
    upload_folder = os.path.join(app.root_path, 'uploads', 'user-uploads')
    print(f"Attempting to serve from: {upload_folder}")  # Debug print statement
    try:
        return send_from_directory(upload_folder, safe_filename)
    except Exception as e:
        return str(e), 404

@app.route('/display/pred/<filename>')
def display_pred_image(filename):
    safe_filename = secure_filename(filename)
    result_folder = os.path.join(app.root_path, 'uploads', 'results')
    print(f"Attempting to serve from: {result_folder}")  # Debug print statement
    try:
        return send_from_directory(result_folder, safe_filename)
    except Exception as e:
        return str(e), 404

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    filename = data['filename']
    confidence_threshold = float(data['confidenceThreshold']) / 100.0  # Convert percentage to a float (if needed)
    upload_folder = os.path.join(app.root_path, 'uploads', 'user-uploads')
    image_path = os.path.join(upload_folder, filename)  # Ensure this path is correctly set
    model_here = Yolov8(pred_img_paths, results_path, confidence_threshold)
    predicted_image = model_here.predict_single_image(image_path)
    model_here.save_output_file(image_path, predicted_image)
    # Handle result_image accordingly, perhaps saving it or updating a database
    return jsonify({'status': 'success', 'message': 'Prediction completed'})
    

@app.route('/clear_data', methods=['POST'])
def clear_data():
    clear_folder(upload_img_folder)
    clear_folder(results_path)
    flash('Data cleared successfully!')
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8888)       