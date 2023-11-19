import os
from typing import List, Union
import dill
from ultralytics.engine.results import Results
from src.util.utils import latest_version_file

class Detections:
    """
    A class for storing and manipulating object detection results.

    Attributes:
    -----------
    bbox : list
        A list of bounding boxes for each detected object in the format [x, y, w, h].
    conf : list
        A list of confidence scores for each detected object.
    id : list
        A list of IDs for each detected object.
    cls : list
        A list of class indices for each detected object.
    boxes : list
        A list of detected objects sorted in descending order of confidence.

    Methods:
    --------
    __init__(self, conf_threshold=0.5, classes=[], class_names=[])
        Initializes the Detections object with the given parameters.
    getlen(self)
        Returns the number of detected objects.
    get(self)
        Returns the bounding boxes, confidence scores, IDs, and class indices for all detected objects.
    getitem(self, index)
        Returns the bounding box, confidence score, ID, and class index for the object at the given index.
    get_boxes(self)
        Returns the list of detected objects sorted in descending order of confidence.
    append(self, bbox, conf, id, cls)
        Appends a detected object to the Detections object.
    correct(self, index, bbox=None, conf=None, id=None, cls=None)
        Corrects the bounding box, confidence score, ID, and/or class index for the object at the given index.
    clear(self)
        Clears all detected objects from the Detections object.
    extract_box(self, box)
        Extracts the bounding box, confidence score, ID, class index, and class name for the given detected object.
    append_boxes(self, results, get_max=False)
        Appends all detected objects in the given results to the Detections object.
    append_detections(self, results, get_max=False)
        Appends all detected objects in the given results to the Detections object and extracts their bounding boxes, confidence scores, IDs, and class indices.
    plot(self, frame, verbose=False)
        Draws bounding boxes and labels for all detected objects on the given frame and returns the annotated frame.
    """

    def __init__(self, conf_threshold:float=0.5, classes:List[int]=[], class_names:List[str]=[]) -> None:
        # attributes
        self.bboxes:List[List[int]] = []
        self.conf:List[float] = []
        self.idx:List[int] = []
        self.names:List[str] = []

        # parameters
        self.conf_threshold:float = conf_threshold
        self.class_names:List[str] = class_names
        self.classes:List[int] = classes
        

    # get methods
    def getLen(self) -> int:
        return len(self.bboxes)
    
    def get(self) -> List:
        return self.bboxes, self.conf, self.idx, self.names
    
    def getIdentity(self, index) -> List:  
        if index < len(self.bboxes):
            return self.bboxes[index], self.conf[index], self.idx[index], self.names[index]
        else:
            return None, None, None, None

    
    # data-adjustment methods
    def append(self, bbox:List[int]=None, conf:float=None, id:int=None, name:str=None) -> None:
        self.bboxes.append(bbox)
        self.conf.append(conf)
        self.idx.append(id)
        self.names.append(name)

    def correct(self, index:int, bbox:List[int]=None, conf:float=None, id:int=None, name:str=None):
        if bbox: self.bboxes[index] = bbox
        if conf: self.conf[index] = conf
        if id: self.idx[index] = id
        if name: self.names[index] = name

    def clear(self) -> None:
        self.bboxes = []
        self.conf = []
        self.idx = []
        self.names = []

    def concat(self, detections:'Detections') -> None:
        if detections == None or detections.getLen() == 0:
            return
        if (self.conf_threshold == None or\
            self.class_names == [] or\
            self.classes == []):
            self.conf_threshold = detections.conf_threshold
            self.class_names = detections.class_names
            self.classes = detections.classes
        elif (detections.conf_threshold != self.conf_threshold or\
            detections.class_names != self.class_names or\
            detections.classes != self.classes):
            raise Exception('Detections object is not compatible')
        self.bboxes.extend(detections.bboxes)
        self.conf.extend(detections.conf)
        self.idx.extend(detections.idx)
        self.names.extend(detections.names)
    
    def extract_box(self, box:Results) -> List:
        x1, y1, x2, y2 = box.xyxy[0]
        xyxy = [int(x1), int(y1), int(x2), int(y2)]
        if box.id is None:
            id = -1
        else:   
            id = int(box.id[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        name = self.class_names[cls]
        return xyxy, conf, id, name, cls

    def append_box(self, box:Results) -> Union[List, None]:
        bbox, conf, id, name, cls = self.extract_box(box)
        # print(bbox, conf, id, name, cls)
        if conf >= self.conf_threshold and cls in self.classes:
            self.append(bbox, conf, id, name)
            return (bbox, conf, id, name)
        else:
            return None
        
    # save and load methods
    def save(self, path:str=None) -> None:
        if path == None:
            path = './detections'
            _, next_file = latest_version_file(path, '.pkl')
            path = os.path.join(path, next_file)
        with open(path, 'wb') as f:
            dill.dump(self, f)
        print('Detections saved to:', '\033[1m' + path + '\033[0m')
    
    def load(self, path:str) -> 'Detections':
        with open(path, 'rb') as f:
            self = dill.load(f)
        return self
    

    
        


