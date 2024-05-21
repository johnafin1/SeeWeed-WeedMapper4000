import os
import json
import logging
from predict import Yolov8
from utils.utils import check_hardware, get_weight_path, get_splitter, get_user_uploads_paths, get_pred_folder_path, get_map_data_folder_path, get_prediction_data_folder_path, get_flight_folder_path, ALLOWED_MOVIE_EXTENSIONS, ALLOWED_IMAGE_EXTENSIONS, ALLOWED_EXTENSIONS
from glob import glob
import pandas as pd

class Flight:
    def __init__(self, name, video_path, telemetry_path=None, map_generated=None, confidence_thres=0.5, iou_thres=0.5, prediction_done=False):
        self.name = name
        self.video_path = video_path
        self.telemetry_path = telemetry_path
        self.map_generated = map_generated
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.prediction_done = prediction_done
        self.flight_path = self.get_user_flight_data()

        self.filename = os.path.basename(video_path)
        base_name, extension = os.path.splitext(self.filename)
        self.processed_filename = f"{base_name}_processed{extension}"
        self.processed_video_path = os.path.join(get_pred_folder_path(), self.processed_filename)
        
        pred_img_paths_mp4 = glob('./uploads/user-uploads/*.mp4')
        pred_img_paths_mov = glob('./uploads/user-uploads/*.mov')
        self.pred_img_paths = pred_img_paths_mp4 + pred_img_paths_mov
        self.model=Yolov8(self.pred_img_paths, get_pred_folder_path(), self.confidence_thres)

        self.prediction_data_file_name = f"{base_name}_prediction_data.json"

    @classmethod
    def create_new(cls, name, video_path, telemetry_path=None, map_path=None, confidence_thres=0.5, iou_thres=0.5):
        if not cls.is_unique_name(name):
            raise ValueError(f"The name '{name}' is already in use. Please choose a different name.")
        flight = cls(name, video_path, telemetry_path, map_path, confidence_thres, iou_thres)
        flight.save_flight()  # Save only when creating new to avoid overwrite on load
        return flight

    @classmethod
    def load_from_json(cls, flight_data):
        return cls(**flight_data)
    
    @staticmethod
    def get_flight_by_name(name):
        # Define the directory where flight data is stored
        directory = get_flight_folder_path()
        
        # Look for a JSON file that matches the flight name
        for filename in os.listdir(directory):
            if filename.endswith('.json') and filename.startswith(name):
                # Construct the full path to the file
                file_path = os.path.join(directory, filename)
                
                # Open and read the JSON file
                with open(file_path, 'r') as file:
                    flight_data = json.load(file)
                
                # Load the flight data into a Flight object
                return Flight.load_from_json(flight_data)
        
        # If no matching flight is found, return None or raise an exception
        return None

    @staticmethod
    def is_unique_name(name):
        """Check if the flight name is unique in the storage directory."""
        for filename in os.listdir(get_flight_folder_path()):
            if filename == f"{name}.json":
                return False
        return True

    def get_user_flight_data(self):
        """Generate and return the path for storing flight data."""
        return os.path.join(get_flight_folder_path(), f"{self.name}.json")

    def add_telemetry(self, telemetry_path):
        """Assign a telemetry path to the flight."""
        self.telemetry_path = telemetry_path
        self.save_flight()  # Update state after adding telemetry

    def run_detection(self):
        """Run video detection and update result path."""
        try:
            self.model=Yolov8(self.pred_img_paths, get_pred_folder_path(), self.confidence_thres)
            frames_data, video_data = self.model.prediction_video(self.video_path)
            video_data.append(frames_data)
            self.result_path = self.create_json_file(video_data)  # Save the data to JSON and get the path
            self.prediction_done = True
            logging.info(f"Detection completed for {self.name}, results saved to {self.result_path}")
        except Exception as e:
            logging.error(f"Failed to process video for {self.name}: {e}")
        finally:
            self.save_flight()

    def load_and_filter_telemetry(self):
        # Load the telemetry data from a CSV file
        df = pd.read_csv(self.telemetry_path)

        # Filter rows where isVideo is 1
        filtered_df = df[df['isVideo'] == 1]

        # Extract required columns
        telemetry_data = filtered_df[['time(millisecond)', 'latitude', 'longitude']]
        return telemetry_data
    
    def load_prediction_data(self):
        # Assuming prediction data is stored as JSON in a known directory
        prediction_data_path = os.path.join('uploads', 'prediction_data', self.prediction_data_file_name)
        try:
            with open(prediction_data_path, 'r') as file:
                prediction_data = json.load(file)
            # Access the second item in the list, which contains the frame data
            frame_data = prediction_data[1]  # Assuming the first item is the summary and the second is the detailed frame data
            return frame_data
        except FileNotFoundError:
            logging.error(f"No prediction data found for {self.name}.")
            return []
        except Exception as e:
            logging.error(f"Failed to load prediction data for {self.name}: {e}")
            return []

    def integrate_telemetry_with_predictions(self, telemetry_data, detection_data):
    # Assume predictions is a list of dicts with keys: frame_number, detections
    # telemetry_data is a DataFrame from the previous step
    
        results = []
        
        # Convert telemetry timestamps to a close approximation of frame numbers if necessary
        # This is an example and might need adjustment based on your specific video framerate and telemetry timing
        telemetry_data['frame_number'] = (telemetry_data['time(millisecond)'] / 1000).astype(int) #need to replace with framerate variable todo
        
        for prediction in detection_data:
            try:
                frame_number = prediction['frame_number']
                detection_count = prediction['detection_count']
                matching_telemetry = telemetry_data[telemetry_data['frame_number'] == frame_number]
            except Exception as e:
                return results
            
            if not matching_telemetry.empty:
                lat = matching_telemetry['latitude'].values[0]
                lon = matching_telemetry['longitude'].values[0]
                
                # Append detection data along with matched telemetry
                result = {
                    'frame_number': frame_number,
                    'latitude': lat,
                    'longitude': lon,
                    'detections': detection_count,
                    #'detection_specs': prediction['detections']
                }
                results.append(result)
    
        return results
    
    def load_map_data(self):
        """Loads map data from a JSON file associated with this flight."""
        try:
            map_data_path = self.get_map_data_path()
            if os.path.exists(map_data_path):
                with open(map_data_path, 'r') as file:
                    map_data = json.load(file)
                return map_data
            else:
                logging.error(f"No map data file found for {self.name}.")
                return []  # Return an empty list if no data file exists
        except Exception as e:
            logging.error(f"Failed to load map data for {self.name}: {str(e)}")
            return []  # Return an empty list in case of an error

    def get_map_data_path(self):
        map_data_folder = get_map_data_folder_path()
        os.makedirs(map_data_folder, exist_ok=True)

        # Create the full path to the map data JSON file
        map_data_file_path = os.path.join(map_data_folder, f"{self.filename}_map_data.json")
        return map_data_file_path
  
    def generate_map(self):
        # Assume detection and telemetry data are stored in a structured way
        telemetry_data = self.load_and_filter_telemetry()
        detection_data = self.load_prediction_data()

        # Process data to integrate detection points with GPS coordinates from telemetry
        map_data = self.integrate_telemetry_with_predictions(telemetry_data, detection_data)
        # Save to a JSON file
        
        map_data_folder = get_map_data_folder_path()
        os.makedirs(map_data_folder, exist_ok=True)

        # Create the full path to the map data JSON file
        map_data_file_path = self.get_map_data_path()
        with open(map_data_file_path, 'w') as file:
            json.dump(map_data, file, indent=4)
        self.map_generated = True
        self.save_flight()

        

    def create_json_file(self, frames_data):
        """Create a JSON file from the detection data and return the file path."""
        json_filename = f"{os.path.splitext(os.path.basename(self.video_path))[0]}_prediction_data.json"
        json_path = os.path.join(get_prediction_data_folder_path(), json_filename)
        try:
            with open(json_path, 'w') as f:
                json.dump(frames_data, f, indent=4)
            logging.info(f"Prediction data saved to {json_path}")
        except IOError as e:
            logging.error(f"Failed to write prediction data to {json_path}: {e}")
            raise
        return json_path

    def save_flight(self):
        """Saves or updates the flight's details in a JSON file."""
        details = {
            "name": self.name,
            "video_path": self.video_path,
            "telemetry_path": self.telemetry_path,
            "map_generated": self.map_generated,
            "prediction_done": self.prediction_done
        }

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.flight_path), exist_ok=True)
            # Write the details to a JSON file
            with open(self.flight_path, 'w') as file:
                json.dump(details, file, indent=4)
            logging.info(f"Flight details saved for {self.name}")
        except Exception as e:
            logging.error(f"Error saving flight details for {self.name}: {e}")

        pass