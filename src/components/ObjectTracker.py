import numpy as np
import math
import cv2
import cvzone
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.components.config import *
from src.utils import *
from src.exception import CustomException   
from src.logger import logging


class ObjectTracker():
    def __init__(self, max_age=20, max_iou_distance=0.7, classes=[], classes_names=[]):
        self.tracker = DeepSort(max_age=max_age, max_iou_distance=max_iou_distance)
        self.tracks:list
        self.detections:list
        if classes == []:
            self.classes = classes_names
        else:      
            self.classes = classes


    def tracking(self, detections, frame):
        try:
            self.tracks = self.tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            loggingInfo(f'Error in ObjectTracker.tracking() \n {e}')
            raise CustomException(e)


    def raw_annotate(self, frame, verbose=False):
        try:
            for track in self.tracks:

                track_id = track.track_id
                bbox = track.to_ltrb()
                curClass = track.get_det_class()
                
                if verbose:
                    print('ID: ',track_id, bbox)

                x, y, w, h = bbox

                cv2.rectangle(frame, (int(x),int(y)),(int(w),int(h)),(0,0,255),2)
                cvzone.putTextRect(frame,  "ID: " + str(track_id), (int(bbox[0]),int(bbox[1]-10)), 
                            scale=1.7, thickness=2, colorT=COLORT,
                            colorR=COLORR, font=cv2.FONT_HERSHEY_PLAIN,
                            offset=2, border=None, colorB=COLORB)
        except Exception as e:
            loggingInfo(f'Error in ObjectTracker.raw_annotate() \n {e}')
            raise CustomException(e)
            

    def match(self, detections, match_iou=0.5):
        try:
            from collections import defaultdict
            class_map = defaultdict(list)
        
            for track in self.tracks:
                class_map[track.get_det_class()].append(track)

            for bbox, conf, detClass in detections:
                for track in class_map[detClass]:
                    iou = IOU_tlwh(track.to_tlwh(), bbox)

                    if iou > match_iou:
                        track_id = track.track_id
                        self.detections.append((track_id, bbox, conf, detClass))
                        class_map[detClass].remove(track)
                        break
        except Exception as e:
            loggingInfo(f'Error in ObjectTracker.match() \n {e}')
            raise CustomException(e)


    def match_annotate(self, frame, verbose=False):
        try:
            for det in self.detections:
                track_id, bbox, conf, curClass = det
                x, y, w, h = bbox

                cv2.rectangle(frame, (int(x), int(y), int(w), int(h)), (0,0,255), 2)
                
                cv2.putText(frame, str(curClass) + " " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # cvzone.putTextRect(frame, str(curClass) + " " + str(track_id), (int(bbox[0]),int(bbox[1]-10)), 
                #             scale=1.7, thickness=2, colorT=COLORT,
                #             colorR=COLORR, font=cv2.FONT_HERSHEY_PLAIN,
                #             offset=2, border=None, colorB=COLORB)
                
                if verbose:
                    print('ID: ',track_id, bbox)
        except Exception as e:
            loggingInfo(f'Error in ObjectTracker.match_annotate() \n {e}')
            raise CustomException(e)

    def Tracker(self, detections, frame, verbose=False, match=True, match_iou=0.5):
        try:
            self.tracking(detections, frame)

            # for track in self.tracks:
            #     print(track.track_id, track.get_det_class(), track.to_tlwh())
            # for det in detections:
            #     print(det)

            self.detections = []
            if match:
                self.match(detections, match_iou)
                self.match_annotate(frame, verbose=verbose)
            else:
                self.raw_annotate(frame, verbose=verbose)
                
        except Exception as e:
            loggingInfo(f'Error in ObjectTracker.Tracker() \n {e}')
            raise CustomException(e)
        
    




    