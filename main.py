import cv2
import numpy as np
import sys
import ast
from ultralytics import YOLO
from src.components.ObjectDetector import ObjectDetector
from src.components.ObjectTracker import ObjectTracker
import pandas as pd
import os
import pafy
from src.logger import logging
from src.exception import CustomException
from src.components.config import *
from src.utils import *

def parse_opt():
    import argparse
    parser = argparse.ArgumentParser()

    def list_of_strings(arg):
        return arg.split(',')

    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH, help='Path to the object detection model')
    parser.add_argument('--classes', type=list_of_strings, default=None, help='Classes for detection')
    parser.add_argument('--path', type=str, default='src//assets/videos//car-0.mp4', help='Path to the video')
    parser.add_argument('--stream', type=bool, default=False, help='Whether to stream from Youtube or not')
    parser.add_argument('--url', type=str, default='https://www.youtube.com/watch?v=vpZBZnolf1U&t=348s', 
                        help='URL of the streaming Youtube video')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--mask', type=str, default='src/assets/images/mask-default.png', help='Path to the mask')
    parser.add_argument('--verbose', action='store_true', help='Whether to print the output or not')
    parser.add_argument('--track', action='store_true', help='Whether to track the objects or not')
    parser.add_argument('--match', action='store_true', help='Whether to annotate the output using Tracker or not')

    return parser.parse_args()

def validate_parse(opt):
    if not os.path.exists(opt.weights):
        loggingInfo(f'Path {opt.weights} does not exist')
        raise ValueError(f'Path {opt.weights} does not exist')
    if not os.path.exists(opt.path) and not opt.stream:
        loggingInfo(f'Path {opt.path} does not exist')
        raise ValueError(f'Path {opt.path} does not exist')
    if not os.path.exists(opt.mask):
        raise ValueError(f'Path {opt.mask} does not exist')

try:
    # load and validate arguments
    opt = parse_opt()
    validate_parse(opt)
    
    # initialize object model and video
    loggingInfo('Loading model...')
    print(opt.weights)
    model = YOLO(opt.weights)
    class_names = list(model.names.values())
    if opt.classes == [] or opt.classes == None:
        classes = class_names
    else:
        classes = opt.classes

    path = opt.path
    winname = 'Frame'
    if not opt.stream:
        loggingInfo('Loading video...')
        if not os.path.exists(path):
            raise ValueError(f'Path {path} does not exist')
        cap = cv2.VideoCapture(f"{path}")
    else:
        loggingInfo('Loading URL...')
        url = opt.url
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # Get the frame rate of the video
    print("Frame rate is: ",frame_rate) 
    # Screen size
    scr_width = SCR_WIDTH
    cap.set(3, scr_width)
    real_width = cap.get(3)
    real_height = cap.get(4)
    aspect_ratio = real_width / real_height
    scr_height = int(scr_width / aspect_ratio)
    cap.set(4, scr_height)

    # masking   
    loggingInfo('Loading mask...')
    mask_path = opt.mask
    mask = np.empty_like(cap.read()[1])
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
    else:
        loggingInfo(f'Warning: Mask {mask_path} does not exist')

    # initialize object detector and tracker
    loggingInfo('Initializing object detector and tracker...')
    detector = ObjectDetector(model=model, classes=classes, classes_names=class_names, conf_threshold=opt.conf)
    if opt.track:
        detect_annotate = False
        tracker = ObjectTracker(max_age=20, max_iou_distance=0.7, classes=classes, classes_names=class_names)
    else:
        detect_annotate = True

    

    print('\n<<Monitor summary>>')
    if opt.stream: 
        print(f'Video URL: {url}')
    else:
        print(f'Video path: {path}')
    print(f'Object detection model: {opt.weights}')
    print(f'Classes: {classes}')
    print(f'Confidence threshold: {opt.conf}')
    print(f'Track: {opt.track}; Match: {opt.match}')
    # print(f'Class names: {class_names}')

    print("""\n<<Starting the video>> (Press Esc to exit)""")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # detecting objects    
        detector.Detector(frame, mask, verbose=opt.verbose, annotate=detect_annotate)
        
        # tracking objects
        if opt.track:
           tracker.Tracker(detector.detections, frame, verbose=opt.verbose, match=opt.match, match_iou=0.5)

        # Name and resize
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, scr_width, scr_height)

        # Show window behind the bounding boxes
        cv2.imshow(winname, frame)
        print('-'*100)
        
        # Set mouse callback
        cv2.setMouseCallback(winname, lambda event, x, y, flags, param: get_coordinates(event, x, y, frame))
        

        # Wait for key press
        delay = int(1000 / frame_rate)
        key = cv2.waitKey(delay)

        # break if esc is pressed
        if key == 27:
            break
except Exception as e:
    loggingInfo(f'Error in main.py \n {e}')
    raise CustomException(e, sys)