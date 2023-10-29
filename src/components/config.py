from ultralytics import YOLO

VIDEO_EXT = ['.mp4', '.avi', '.mov', '.mkv']

# Limit line for counter
LINE_BEGIN = {
    'car-0': [[0,560,1920,560]],
    'car-1': [[200,587,1030,587]],
    'car-2': [[135,435,1190,435]],
    'car-3': [],
    'car': [],
}

CONF_THRESHOLD = 0.5
WEIGHTS_PATH = f'yolov8/weights/yolov8l.pt'
DEFAULT_URL = 'https://www.youtube.com/watch?v=Jsn8D3aC840&list=PL1FZnkj4ad1PFJTjW4mWpHZhzgJinkNV0&index=17'

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



