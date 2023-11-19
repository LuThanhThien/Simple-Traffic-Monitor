from ultralytics import YOLO
from src.modules.utils import *
import os
import re
import os


VIDEO_EXT = ['.mp4', '.avi', '.mov', '.mkv']

# Limit line for counter
LINE_BEGIN = {
    'car-0': [[0,560,1920,560]],
    'car-1': [[200,587,1030,587]],
    'car-2': [[135,435,1190,435]],
    'car-3': [],
    'car': [],
} 

# Screen size
SCR_WIDTH = 1200

# display boxes
COLORB = (0, 255, 0)
COLORT = (255, 255, 255)

COLORR = (0, 255, 0)
COLORC = (0, 255, 0)

TEXTRECT_COLORB = (0, 255, 0)
TEXTRECT_COLORT = (255, 255, 255)

CONRECT_COLORR = (0, 255, 0)
CONRECT_COLORC = (0, 255, 0)


# DETECT
# CLASSES = ['car', 'motorcycle', 'truck', 'bus', 'bicycle']
# FOCUS_CLASSES = ['car', 'motorcycle', 'truck', 'bus']
CLASSES = [1, 2, 3, 5, 7]
FOCUS_CLASSES = [2, 3, 5, 7]
CONF_THRESHOLD = 0.5
WEIGHTS_PATH = f'yolov8/weights/yolov8m.pt'
DEFAULT_URL = 'https://www.youtube.com/watch?v=Jsn8D3aC840&list=PL1FZnkj4ad1PFJTjW4mWpHZhzgJinkNV0&index=17'

# plate detection
weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'YOLOSH', 'runs')
lastest_ocr, next_ocr = latest_version_file(weight_path, name='yolocr-v')
lastest_plate, next_ocr = latest_version_file(weight_path, name='yoloplate-v')

PLATE_WEIGHTS = os.path.join(weight_path, lastest_plate, 'weights', 'best.pt')
OCR_WEIGHTS = os.path.join(weight_path, lastest_ocr, 'weights', 'best.pt')


CHAR2INT = {
    'O': '0',
    '1': 'I',
    '2': 'Z',
    '3': 'J',
    '8': 'B',
    '6': 'G',
    '7': 'T',
    '4': 'A',
    '5': 'S',}

INT2CHAR = {
    '0': 'O', 
    'I': '1',
    'L': '1',
    'Z': '2',
    'B': '3',
    'G': '6',
    'T': '7',
    'A': '4',
    'S': '5',}

DICT_OCR_CLASSES = {
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4',
                5: '5',
                6: '6',
                7: '7',
                8: '8',
                9: '9',
                10: 'A',
                11: 'B',
                12: 'C',
                13: 'D',
                14: 'E',
                15: 'F',
                16: 'G',
                17: 'H',
                18: 'I',
                19: 'J',
                20: 'K',
                21: 'L',
                22: 'M',
                23: 'N',
                24: 'O',
                25: 'P',
                26: 'Q',
                27: 'R',
                28: 'S',
                29: 'T',
                30: 'U',
                31: 'V',
                32: 'W',
                33: 'X',
                34: 'Y',
                35: 'Z',
            }


if __name__ == '__main__':
    print(lastest_ocr)
    print(lastest_plate)