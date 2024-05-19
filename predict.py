import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch
from utils.utils import check_hardware, get_weight_path, get_pred_folder_path, get_splitter, get_user_uploads_paths, get_map_data_folder_path, get_prediction_data_folder_path
from utils.logger import PythonLogger 
logger = PythonLogger().logger 
import os
import time
import json

class Yolov8:

    def __init__(self, image_paths, results_path, confidence_thres=0.3, iou_thres=0.5):
        """
        Initializes an instance of the Yolov8 class.

        Args:
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.video_framerate_offset = 10
        self.image_paths = image_paths
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.results_path = results_path
        self.predicted_images = []
        self.prediction_times = []
        self.detection_count = []
        self.prediction_time = 0
        # Load the class names from the COCO dataset
        self.classes = {0: 'Mysore-Thorn'}
        self.json_data = []  # Initialize an empty list to store detection data
        self.json_file_path = os.path.join(results_path, "detections.json")  # Define the JSON file path


        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def clear(self):
        self.predicted_images = []
        self.prediction_times = []
        self.detection_count = []
        self.prediction_time = 0

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = (255, 0, 0)

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def image_path_to_img(self, image_path):
        # Read the input image using OpenCV
        self.img = cv2.imread(image_path)
        return self.preprocess_image(self.img)

    def preprocess_image(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def preprocess_video(self, img):
        # Convert the image color space from BGR to RGB
        self.img_height, self.img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize the image
        # Ensure you set self.input_width and self.input_height properly
        img = cv2.resize(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def process_video(self, video_path):
        #Change this value if your pc is struggling
        self.video_framerate_offset = 10
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        # Read the first frame to get the video resolution
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read the first frame from the video.")
            return
        
        # Get the dimensions of the frame
        height, width = frame.shape[:2]
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Reset the capture to the start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
        # Assuming self.input_width and self.input_height are set correctly
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'mp4v' for MP4
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        extension = os.path.splitext(video_path)[1]
        output_file_name = f'{base_name}_processed{extension}'
        output_file_path = os.path.join('uploads/results', output_file_name)
        out = cv2.VideoWriter(output_file_path, fourcc, fps/self.video_framerate_offset, (width, height))

        logger.info(f"Frame HELLO Width={self.input_width}: Height={self.input_height}")

        frames_data = []
        frame_count = 0
        processed_frame_count = 0
        total_prediction_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Only process every 30th frame
            if frame_count % self.video_framerate_offset == 0:
                prediction_time, detection_count, detections = self.process_frame(frame, frame_count, out, base_name, total_prediction_time)
                processed_frame_count += 1
                total_prediction_time += prediction_time
                logger.info(f"Frame: {processed_frame_count}\nTime to predict: {prediction_time}")
                frames_data.append({
                    "frame_number": processed_frame_count,
                    "inference_time": prediction_time,
                    "detection_count": detection_count,  # assuming detections is a list of dictionaries with details
                    "detections": detections  # assuming detections is a list of dictionaries with details
                })

        cap.release()
        out.release()

        avg_prediction_time = total_prediction_time / processed_frame_count if processed_frame_count else 0
        video_data = []
        video_data.append({
            "framerate_offset" : self.video_framerate_offset,
            "frames_processed" : frame_count,
            "avg_prediction_time" : avg_prediction_time,
            "total_prediction_time" : total_prediction_time,
            "base_name" : base_name,
            "extension" : extension
        })
        logger.info(f"Video processing completed. Frames processed: {frame_count}. Average prediction time per frame: {avg_prediction_time:.4f} seconds.")
        return frames_data, video_data

    def process_frame(self, frame, frame_count, out, base_name, total_prediction_time):
        img_data = self.preprocess_video(frame)
        height, width, channels = frame.shape
        logger.info(f"Processing frame {frame_count}: Width={width}, Height={height}, Channels={channels}")
    
        start_time = time.time()
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        prediction_time = time.time() - start_time

        total_prediction_time += prediction_time

        processed_frame, detection_count, detections = self.postprocess(frame, outputs, base_name, frame_count)

        height_p, width_p, channels_p = processed_frame.shape
        logger.info(f"Processing frame {frame_count}: Width={width_p}, Height={height_p}, Channels={channels_p}")

         # Write the processed frame
        out.write(processed_frame)

        return prediction_time, detection_count, detections

    def postprocess(self, input_image, output, video_name = None, frame_count = None):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
        
        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model. Search this for further information from the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        detections = []
        if(frame_count and video_name):
            none = None

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        image_count = 0
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Collect each detection's details
            detections.append({
                "class_id": int(class_id),
                "confidence": float(score),
                "bbox": box
            })
            
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)
        self.detection_count.append(len(indices))
        # Return the modified input image
        return input_image, len(indices), detections

    def save_output(self):
        pred_folder_path = self.results_path
        os.makedirs(pred_folder_path, exist_ok=True)

        for path, pred_arr in zip(self.image_paths, self.predicted_images):
            img_name = pred_folder_path + '/' + path.split(get_splitter())[-1]
            cv2.imwrite(img_name, pred_arr)
            logger.info(f"Saved image at: {img_name}")

        print()
        logger.info(f"All predicted images have been saved!")
        logger.info(f"Total prediction time: {self.prediction_time}s")
        
    def save_output_file(self, path, predicted_image):
        pred_folder_path = self.results_path
        os.makedirs(pred_folder_path, exist_ok=True)

        # Construct the output file name based on the original file's name
        img_name = os.path.join(pred_folder_path, os.path.basename(path))

        # Save the predicted image to disk
        cv2.imwrite(img_name, predicted_image)
        logger.info(f"Saved image at: {img_name}")

        # Log completion and any relevant statistics
        print()
        logger.info("Predicted image has been saved!")
        logger.info(f"Total prediction time: {self.prediction_time}s")


    def get_image_data(self):
        # Assuming that the processed images are saved with the same filename but in a different directory
        pred_folder_path = self.results_path
        image_data = []
        i = 0
        for original_path in self.image_paths:
            if not original_path.lower().endswith(('.mp4', '.avi', '.MOV', '.mov')):  # Extend this list as needed
                filename = original_path.split(get_splitter())[-1]
                processed_path = pred_folder_path + filename

                # For now, let's assume a constant confidence score of 0.95 for all images
                # You should replace this with the actual confidence score from your model
                prediction_times = self.prediction_times[i]
                detection_count = self.detection_count[i]

                image_data.append((original_path, processed_path, prediction_times, detection_count))
                i += 1

            else:
                filename = original_path.split(get_splitter())[-1]
                processed_path = pred_folder_path + 'processed_video.mp4'
                original_path = get_user_uploads_paths() + filename
                image_data.append((original_path, processed_path, 'unknown', 'unknown'))

        for data in image_data:
            print(data)

        return image_data
    
    def prediction_video(self, video_path):
        weight_path = get_weight_path()
        cuda = check_hardware()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weight_path, providers=providers)

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        frames_data = None
        if video_path.lower().endswith(('.mp4', '.avi', '.MOV', '.mov')):  # Extend this list as needed
                logger.info(f"Processing video: {video_path}")
                frames_data, video_data = self.process_video(video_path)
        return frames_data, video_data

    def predict_single_image(self, image_path):
        weight_path = get_weight_path()
        cuda = check_hardware()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weight_path, providers=providers)

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        if image_path.lower().endswith(('.mp4', '.avi', '.MOV', '.mov')):  # Extend this list as needed
                logger.info(f"Processing video: {image_path}")
                self.process_video(image_path)
        else:
            # Preprocess the image data
            image_name = image_path.split(get_splitter())[-1]
            print()
            logger.info(f"Preprocessing {image_name}")
            image_path=image_path
            img_data = self.image_path_to_img(image_path)
            #img_data = self.preprocess(image_path=path)

            # Run inference using the preprocessed image data
            logger.info(f"Predicting ...")
            start_time = time.time()
            outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
            total_time = round(time.time() - start_time, 3)
            logger.info(f"Predictiont time: {total_time}s")
            self.prediction_time += total_time
            self.prediction_times.append(total_time)

            # Perform post-processing on the outputs to obtain output image.
            predicted_image, detection_count, detections = self.postprocess(self.img, outputs)  # output image
            
            self.predicted_images.append(predicted_image)
            stats = self.make_prediction_stat(image_name, total_time, detection_count, detections)
            self.save_prediction_stats(image_name, stats)
            return predicted_image

    def predict(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        weight_path = get_weight_path()
        cuda = check_hardware()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weight_path, providers=providers)

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        for path in self.image_paths:
            if path.lower().endswith(('.mp4', '.avi', '.MOV', '.mov')):  # Extend this list as needed
                logger.info(f"Processing video: {path}")
                self.process_video(path)
            else:
                # Preprocess the image data
                image_name = path.split(get_splitter())[-1]
                print()
                logger.info(f"Preprocessing {image_name}")
                image_path=path
                img_data = self.image_path_to_img(image_path)
                #img_data = self.preprocess(image_path=path)

                # Run inference using the preprocessed image data
                logger.info(f"Predicting ...")
                start_time = time.time()
                outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
                total_time = round(time.time() - start_time, 3)
                logger.info(f"Predictiont time: {total_time}s")
                self.prediction_time += total_time
                self.prediction_times.append(total_time)

                # Perform post-processing on the outputs to obtain output image.
                predicted_image, detection_count, detections = self.postprocess(self.img, outputs)  # output image

                self.predicted_images.append(predicted_image)

    def make_prediction_stat(self, image_name, prediction_time, detection_count, detections):
        # Check if the stats file exists
        data = []
        # Append new data
        data.append({
            "image": image_name,
            "prediction_time": prediction_time,
            "detection_count": detection_count,
            "detections": detections
        })
        return data
    
    def save_prediction_stats(self, image_name, data):
        json_filename = image_name + '.json'
        stats_path = os.path.join(get_prediction_data_folder_path(), json_filename)

        if os.path.exists(stats_path):
            with open(stats_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        existing_data.extend(data)  # Append new data

        with open(stats_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

        logger.info(f"Saved prediction stats for {image_name}")