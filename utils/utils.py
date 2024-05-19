import torch
import subprocess
import os
import shutil
import numpy as np
from sys import platform 
from utils.logger import PythonLogger 
from flask import flash
logger = PythonLogger().logger 
ALLOWED_MOVIE_EXTENSIONS = {'mp4', 'avi', 'mov', 'MOV'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4', 'MOV', 'mov', 'avi'])
ALLOWED_CSV_EXTENSIONS = {'csv', 'xls', 'xlsx'}

def get_splitter():
    if platform.startswith("linux") or platform.startswith("darwin"):
        return '/'
    else:
        return "\\"

def check_hardware():
    # Need to use torch to check gpu
    # Check whether hardware has gpu, if there is not gpu, model will predict using cpu
    
    cuda_avail = torch.cuda.is_available()
    
    if cuda_avail:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f'GPU Count: {device_count}, GPU Name: {device_name}')
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"GPU Available: {cuda_avail}, Device: {device}")
    flash(f"GPU Available: {cuda_avail}, Device: {device}")
    return cuda_avail

def seed_everything(seed=43):
    '''
      Make PyTorch deterministic.
    '''    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4', 'MOV', 'mov'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS    

def allowed_movie_file(filename):
    ALLOWED_MOVIE_EXTENSIONS = {'mp4', 'avi', 'mov', 'MOV'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MOVIE_EXTENSIONS

def allowed_csv_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

#Paths    

def get_pred_img_paths():
    paths = []
    splitter = get_splitter()
    
    for dirname, _, filenames in os.walk(get_pred_folder_path()):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))

    new_paths = sorted([path.split(splitter)[-1] for path in paths])
    return new_paths      

def get_pred_video_paths():
    paths = []
    splitter = get_splitter()
    
    for dirname, _, filenames in os.walk(get_pred_folder_path()):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))

    new_paths = sorted([path.split(splitter)[-1] for path in paths])
    return new_paths     

def get_pred_folder_path():
    return r"uploads/results/"

def get_weight_path():
    #return r"weights/trial6-best.onnx"
    #return r"weights/best.onnx"
    return r"weights/best-run2.onnx"
    
def get_user_csv_paths():
    path = r'uploads/user-csv/'    
    os.makedirs(path, exist_ok=True)
    return path

def get_user_uploads_paths():
    path = r'uploads/user-uploads/'    
    os.makedirs(path, exist_ok=True)
    return path

def get_map_data_folder_path():
    return r"uploads/flight_data/"

def get_flight_folder_path():
    return r"uploads/flight/"

def get_prediction_data_folder_path():
    return r"uploads/prediction_data/"

def clean_up():
    '''
       Delete
       1: user-upload pdf folder
       2: preprocessed_imgs folder 
    '''

    user_upload_dir = get_user_uploads_paths()
    # this is wrogn preprocess_imgs_dir = get_prediction_data_folder_path()
    pred_imgs_dir = get_pred_folder_path()

    shutil.rmtree(user_upload_dir)
    shutil.rmtree(preprocess_imgs_dir)
    shutil.rmtree(pred_imgs_dir)
