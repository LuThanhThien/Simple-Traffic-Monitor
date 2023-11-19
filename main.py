import cv2
import numpy as np
import argparse
import os
import pafy
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from src.engine.Detector import UltraDetector
from src.engine.Reader import EasyReader
from src.logger import logging
from src.exception import CustomException
from src.config import *
from src.util.utils import *


def parse_opt():
    parser = argparse.ArgumentParser()

    def list_of_strings(arg):
        return arg.split(',')

    parser.add_argument('--weights', type=str, default=WEIGHTS_PATH, help='Path to the object detection model')
    parser.add_argument('--classes', type=list_of_strings, default=CLASSES, help='Classes for detection')
    parser.add_argument('--path', type=str, default='src//assets/videos//test-4.mp4', help='Path to the video')
    parser.add_argument('--stream', action='store_true', help='Whether to stream from Youtube or not')
    parser.add_argument('--url', type=str, default=DEFAULT_URL, help='URL of the streaming Youtube video')
    parser.add_argument('--mask', type=str, default='src/assets/images/mask-default.png', help='Path to the mask')
    parser.add_argument('--verbose', action='store_true', help='Whether to print the output or not')
    parser.add_argument('--wait', action='store_true', help='Whether to wait for key press or not')
    return parser.parse_args()


def validate_parse(opt):
    if not os.path.exists(opt.weights):
        if opt.weights == WEIGHTS_PATH:
            _ = YOLO(opt.weights) # Download the weights
        loggingInfo(f'Path {opt.weights} does not exist')
        raise ValueError(f'Path {opt.weights} does not exist')
    if not os.path.exists(opt.path) and not opt.stream:
        loggingInfo(f'Path {opt.path} does not exist')
        raise ValueError(f'Path {opt.path} does not exist')
    if not os.path.exists(opt.mask):
        raise ValueError(f'Path {opt.mask} does not exist')


def main():
    # Load and validate arguments
    opt = parse_opt()
    validate_parse(opt)

    # Model setup
    loggingInfo('Loading models...')
    print('Loading vehicle detection model from', opt.weights)
    model_vehicle = YOLO(opt.weights)
    print('Loading plate detection model from', PLATE_WEIGHTS)
    model_plate = YOLO(PLATE_WEIGHTS)
    # print('Loading character detection model from', OCR_WEIGHTS)
    # model_ocr = YOLO(OCR_WEIGHTS)
        

    # Video setup
    path = opt.path
    if not opt.stream:
        loggingInfo('Loading media...')
        if not os.path.exists(path):
            raise ValueError(f'Path {path} does not exist')
        cap = cv2.VideoCapture(f"{path}")
    else:
        loggingInfo('Loading URL...')
        url = opt.url
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)


    # Screen setup
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # Get the frame rate of the video
    print("Frame rate is: ",frame_rate) 
    # scr_width = SCR_WIDTH
    # cap.set(3, scr_width)
    # real_width = cap.get(3)
    # real_height = cap.get(4)
    # aspect_ratio = real_width / real_height
    # scr_height = int(scr_width / aspect_ratio)
    # cap.set(4, scr_height)
    winname = 'Frame'

    # Mask setup   
    loggingInfo('Loading mask...')
    mask_path = opt.mask
    mask = np.empty_like(cap.read()[1])
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
    else:
        loggingInfo(f'Warning: Mask {mask_path} does not exist')


    # Initialize object detector and tracker
    loggingInfo('Initializing objects...')
    vehicleObj = UltraDetector(model=model_vehicle, classes=opt.classes)
    plateObj = UltraDetector(model=model_plate)

    print('\n<<Monitor summary>>')
    if opt.stream: 
        print(f'Video URL: {url}')
    else:
        print(f'Video path: {path}')
    print(f'Object detection model: {opt.weights}')
    print(f'Classes: {opt.classes}')
    # print(f'Class names: {class_names}')

    print("""\n<<Starting the video>> (Press Esc to exit)""")

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()

        if not ret:
            break
        
        # Pipeline   
        vehicleObj.Detect(frame, mask, tracker="botsort.yaml")
        plateObj.Detect(frame, tracker="botsort.yaml")
        union_node = vehicleObj.frame_nodes.deepunion(plateObj.frame_nodes)
        union_node.summarize()

        # Name and resize
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(winname, scr_width, scr_height)

        # Show window behind the bounding boxes
        frame = union_node.plot(seperate=False, line_width=2)
        cv2.imshow(winname, frame)
        print(f'Frame took {time.time() - start} seconds\n')
        print('-'*100)
        
        # Set mouse callback
        cv2.setMouseCallback(winname, lambda event, x, y, flags, param: get_coordinates(event, x, y, frame))

        # Wait for key press
        if opt.wait:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)

        # break if esc is pressed
        if key == 27:
            break


if __name__ == '__main__':
    main()