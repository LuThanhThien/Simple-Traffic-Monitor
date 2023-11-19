from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
import time, sys
from typing import List, Union
from src.config import *
from src.exception import CustomException
from src.logger import logging
from src.util.utils import *
from src.util.preprocess import *
from src.util.reader import *
from src.frame.Cascades import FrameTree
import easyocr


class EasyReader:
    def __init__(self, conf_threshold:float=0.5) -> None:
        # models
        start = time.time()
        self.reader = easyocr.Reader(['en'], gpu=True)

        # parameters
        self.ocr_names:list = list(DICT_OCR_CLASSES.values())
        self.ocr_classes:list = list(DICT_OCR_CLASSES.keys())
        self.focus_classess:list = FOCUS_CLASSES
        self.conf_theshold:float= conf_threshold

        # results
        self.ocr = None
        self.frame_nodes:FrameTree
        print(f'{self.__class__.__name__}.__init__() took', time.time() - start, 'seconds\n')
    

    # get all
    def getTree(self):
        return self.frame_nodes

    def saveTree(self, path:str='detections\detect0.pkl'):
        self.frame_nodes.save(path)

    def loadTree(self, path:str):
        self.frame_nodes.load(path)

        
    # def detect_easyocr(self, frame):
    #     self.ocr = []
    #     boxes = self.plates.get_detections()
    #     for i, b in enumerate(boxes):
    #         print('Number plate', i)
    #         bbox = b.bbox
    #         crop_plate = crop(frame, bbox)
    #         sufflix = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    #         # preprocess
    #         preprocess_plate = pipeline(crop_plate, name=f'image-preprocess-{sufflix}', save=True, showf=False, savef=False)
    #         # read 
    #         results = easyocr_predict([preprocess_plate], bbox, self.reader, name=f'image-easyocr-{sufflix}', show=False)
    #         self.ocr.extend(results)

    # def detect_yoloocr(self, frame):
    #     from multiprocessing import Pool
    #     self.ocr = []
    #     preprocessed_plates = []
    #     for i, b in enumerate(self.boxes):
    #         bbox = xywh2xywh(b)
    #         crop_plate = crop(frame, bbox)
    #         sufflix = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    #         # preprocess and apend
    #         preprocessed_plate = pipeline(crop_plate, name=f'image-preprocess-{sufflix}', save=True, showf=False, savef=False)
    #         # preprocessed_plates.extend(preprocessed_plate)
    #         yoloocr_predict(preprocessed_plate, bbox, self.ocr_model, name=f'image-yoloocr-{sufflix}', show=False)
    #     # self.ocr.extend(results)

