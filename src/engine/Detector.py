import numpy as np
import sys, time
import cv2
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from typing import List, Union
from src.exception import CustomException
from src.logger import logging
from src.config import *
from src.util.utils import *
from src.frame.Cascades import FrameTree


class UltraDetector:
    def __init__(self, model:Model, classes:List[int]=[], conf_threshold:float=0.5) -> None:
        # parameters
        start = time.time()
        dict_name:dict = model.names
        self.class_names = list(dict_name.values())
        if classes == [] or classes == None:
            self.classes = list(dict_name.keys())
        else:      
            self.classes = classes
        self.conf_threshold = conf_threshold

        # model
        self.model = model

        # results
        self.tracks:list
        self.frame_nodes:FrameTree
        self.annotate_frame:np.ndarray
        print(f'{self.__class__.__name__}.__init__() took', time.time() - start, 'seconds\n')


    # get all
    def getTree(self):
        return self.frame_nodes
    
    def getNodes(self):
        return self.frame_nodes.getNodes()

    def saveTree(self, path:str='detections\detect0.pkl'):
        self.frame_nodes.save(path)

    def loadTree(self, path:str):
        self.frame_nodes.load(path)
    

    # detect and post process
    def masking(self, disFrame:np.ndarray, mask:np.ndarray):
        if np.all(mask == 0):
            return disFrame
        resize_mask = cv2.resize(mask, (disFrame.shape[1], disFrame.shape[0]))
        maskFrame = cv2.bitwise_and(disFrame, resize_mask)
        return maskFrame
        

    def tracking(self, maskFrame:np.ndarray, **krargs):
        # track
        self.tracks = self.model.track(maskFrame, persist=True, classes=self.classes, **krargs)
        # init frame nodes 
        self.frame_nodes = FrameTree(conf_threshold=self.conf_threshold, classes=self.classes,
                                     class_names=self.class_names, root=maskFrame, name='root', id=-1, parent=None)
        # apped detections
        self.frame_nodes.add(self.tracks)


    def detecting(self, maskFrame:np.ndarray, **krargs):
        # track
        self.tracks = self.model.predict(maskFrame, stream=True, classes=self.classes, **krargs)
        # init frame nodes 
        self.frame_nodes = FrameTree(conf_threshold=self.conf_threshold, classes=self.classes,
                                     class_names=self.class_names, root=maskFrame, name='root', id=-1, parent=None)
        # apped detections
        self.frame_nodes.add(self.tracks)


    def plot(self, verbose:bool=False, **krargs):
        return self.frame_nodes.plot(verbose=verbose, **krargs)
    
    
    # main
    def Track(self, frame:np.ndarray=np.array([]), mask:np.ndarray=np.array([]), **krargs):
        try:
            if len(frame) == 0:
                raise ValueError('Frame is empty.')
            if len(mask) == 0:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            start = time.time()
            maskFrame = self.masking(frame, mask)
            self.tracking(maskFrame, **krargs)
            self.frame_nodes.summarize()
            
            print(f'{self.__class__.__name__}.Detect() took', time.time() - start, 'seconds\n')
        except Exception as e:
            loggingInfo(f'Error in {self.__class__.__name__}.Detect() \n {e}')
            raise CustomException(e, sys)

    def Detect(self, frame:np.ndarray=np.array([]), mask:np.ndarray=np.array([]), **krargs):
        try:
            if len(frame) == 0:
                raise ValueError('Frame is empty.')
            if len(mask) == 0:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            start = time.time()
            maskFrame = self.masking(frame, mask)
            self.detecting(maskFrame, **krargs)
            self.frame_nodes.summarize()
            
            print(f'{self.__class__.__name__}.Detect() took', time.time() - start, 'seconds\n')
        except Exception as e:
            loggingInfo(f'Error in {self.__class__.__name__}.Detect() \n {e}')
            raise CustomException(e, sys)
    