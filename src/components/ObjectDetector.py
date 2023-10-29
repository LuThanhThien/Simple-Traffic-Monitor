from ultralytics import YOLO
import numpy as np
import cvzone
import math
import cv2
import os
from src.components.config import *
from src.exception import CustomException
from src.logger import logging
from src.utils import *



class ObjectDetector:
    def __init__(self, model, classes=[], classes_names=[], conf_threshold=0.5) -> None:
        if classes == [] or classes == None:
            self.classes = classes_names
        else:      
            self.classes = classes
        self.class_names = classes_names
        self.model = model
        self.conf_threshold = conf_threshold
        self.detections:list
        self.results:list
        self.boxes:list

    def masking(self, disFrame, mask):
        try:
            if np.all(mask == 0):
                return disFrame
            resize_mask = cv2.resize(mask, (disFrame.shape[1], disFrame.shape[0]))
            maskFrame = cv2.bitwise_and(disFrame, resize_mask)
            return maskFrame
        except Exception as e:
            loggingInfo(f'Error in ObjectDetector.masking() \n {e}')
            raise CustomException(e)
    
    def detect(self, maskFrame):
        try:
            self.boxes = []
            self.results = self.model(maskFrame, stream=True)
            for r in self.results:
                for box in r.boxes:
                    self.boxes.append(box)    
        except Exception as e:
            loggingInfo(f'Error in ObjectDetector.detect() \n {e}')
            raise CustomException(e)

    
    def annotate(self, disFrame, verbose=False, annotate=True):
        try:
            self.detections = []
            for box in self.boxes:
                xb, yb, w, h = box.xywh[0]
                x1 = int(xb - w / 2)
                y1 = int(yb - h / 2)
                x2 = int(x1 + w) 
                y2 = int(y1 + h) 
            
                # Confidence
                conf = math.ceil(box.conf[0])

                # Class names
                cls = box.cls[0]

                # Get current class
                currentClass = self.class_names[int(cls)]

                # print(f'{currentClass} {conf}%')

                if (currentClass in self.classes) and (conf > self.conf_threshold):
                    # annotate text
                    if annotate:
                        cv2.putText(disFrame, f'{currentClass} {conf*100}%',
                                    (max(0, int(x1)), max(40, int(y1 - 10))),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, COLORT, 2, cv2.LINE_AA)

                        # cvzone.putTextRect(disFrame, f'{currentClass} {conf}%',
                        #                     (max(0, int(x1)), max(40, int(y1 - 10))),
                        #                     scale=1.7, thickness=2, colorT=COLORT,
                        #                     colorR=COLORR, font=cv2.FONT_HERSHEY_PLAIN,
                        #                     offset=2, border=None, colorB=COLORB)

                        # annotate boxes
                        cvzone.cornerRect(disFrame, bbox=(x1, y1, int(w), int(h)), l=10, t=3, rt=3,
                                            colorR=COLORR, colorC=COLORC)
                        
                    if verbose:
                        print(f'{currentClass} {conf*100}%')
                        print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, w: {w}, h: {h}')

                    self.detections.append(([x1, y1, int(w), int(h)], conf, currentClass))

        except Exception as e:
            loggingInfo(f'Error in ObjectDetector.annotate() \n {e}')
            raise CustomException(e)



    def Detector(self, disFrame, mask, verbose=False, annotate=True):
        try:
            maskFrame = self.masking(disFrame, mask)
            self.detect(maskFrame)
            self.annotate(disFrame, verbose=verbose, annotate=annotate)
        except Exception as e:
            loggingInfo(f'Error in ObjectDetector.Detector() \n {e}')
            raise CustomException(e)