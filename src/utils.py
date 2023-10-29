import cv2
import numpy as np
from src.exception import CustomException
from src.logger import logging

def loggingInfo(message, verbose=True):
    try: 
        if verbose: print(f'INFO: {message}')
        return logging.info(message)
    except Exception as e:
        raise CustomException(e)

def get_coordinates(event, x, y, frame):
    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.putText(frame, f'({x}, {y})', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f'[Utils::get_coordinates] Mouse click at: ({x},{y})')
    except Exception as e:
        raise CustomException(e)

def IOU_tlwh(bb_test, bb_gt):
    try:
        x1 = max(bb_test[0], bb_gt[0])
        y1 = max(bb_test[1], bb_gt[1])
        x2 = min(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
        y2 = min(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
        
        # Calculate the area of intersection
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate the area of union
        area_bb_test = bb_test[2] * bb_test[3]
        area_bb_gt = bb_gt[2] * bb_gt[3]
        union = area_bb_test + area_bb_gt - intersection
        
        # Calculate and return the Intersection over Union (IoU)
        return intersection / union
    except Exception as e:
        raise CustomException(e)
