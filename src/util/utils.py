import cv2
from ultralytics.utils.plotting import Annotator
from src.logger import logging
import os, re
import numpy as np
# utils does not contain values import from other modules except logging,
# preventing circular import

def loggingInfo(message, verbose=True): 
    """
    Logs an information message.
    """
    if verbose: print(f'INFO: {message}')
    return logging.info(message)


def latest_version_file(path, name='model', extension=None):
    # Get a list of all files in the folder
    files = os.listdir(path)
    os.makedirs(path, exist_ok=True)    
    pattern = f"^{name}\d+{re.escape(extension)}$" if extension else f"^{name}\d+"
    filtered_files = [f for f in files if re.match(pattern, f)]

    if not files:
        print(f'[Utils::latest_version_file] No file in {path}')
        if not extension:
            print('[Utils::latest_version_file] WARNING: No extension provided')
        return None, f'{name}0.{extension or ""}'
    if not filtered_files:
        print(f'[Utils::latest_version_file] No file with name "{name}" in {path} ')
        return None, f'{name}0.{extension or ""}'

    # versions
    versions = [int(re.search(r'\d+', f).group()) for f in filtered_files]
    max_version = max(versions)
    extension = filtered_files[versions.index(max_version)].split(name+str(max_version))[-1] 

    # return file name
    current_file_name = f"{name}{max_version}{extension}" 
    next_file_name = f"{name}{max_version + 1}{extension}" 
    return current_file_name, next_file_name


def get_coordinates(event, x, y, frame):
    """
    Displays the coordinates of a mouse click on a given frame.

    Args:
        event (int): The type of mouse event that occurred.
        x (int): The x-coordinate of the mouse click.
        y (int): The y-coordinate of the mouse click.
        frame (numpy.ndarray): The frame on which the mouse click occurred.

    Returns:
        None
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.putText(frame, f'({x}, {y})', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(f'[Utils::get_coordinates] Mouse click at: ({x},{y})')


def xywh2xyxy(bbox):
    """
    Converts a bounding box from the format of (top left x, top left y, width, height) to (top left x, top left y, bottom right x, bottom right y).
    
    Args:
        bbox (tuple): The bounding box in the format of (top left x, top left y, width, height).

    Returns:
        list: The bounding box in the format of (top left x, top left y, bottom right x, bottom right y).
    """
    x, y, w, h = bbox
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)
    return [x1, y1, x2, y2]


def IOU(bb_test, bb_gt, format:str='xyxy'):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes in the format of (top left x, top left y, bottom right x, bottom right y).
    
    Args:
        bb_test (tuple): The first bounding box in the format of (top left x, top left y, bottom right x, bottom right y).
        bb_gt (tuple): The second bounding box in the format of (top left x, top left y, bottom right x, bottom right y).
    
    Returns:
        float: The IoU between the two bounding boxes.
    """
    if format not in ['xyxy', 'xywh']:
         raise ValueError('Invalid argument for format. Must be one of [xyxy, xywh]')
    
    if format == 'xywh':
        bb_test = xywh2xyxy(bb_test)
        bb_gt = xywh2xyxy(bb_gt)

    x1 = max(bb_test[0], bb_gt[0])
    y1 = max(bb_test[1], bb_gt[1])
    x2 = min(bb_test[2], bb_gt[2])
    y2 = min(bb_test[3], bb_gt[3])
    
    # Calculate the area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate the area of union
    area_bb_test = bb_test[2] * bb_test[3]
    area_bb_gt = bb_gt[2] * bb_gt[3]
    union = area_bb_test + area_bb_gt - intersection
    
    # Calculate and return the Intersection over Union (IoU)
    return intersection / union


def IOB(bbox1, bbox2, format:str='xyxy'):
    """
    Calculates the Intersection over Bbox (IoB) between two bounding boxes in the format of (top left x, top left y, bottom right x, bottom right y).
    
    Args:
        bbox1 (tuple): The first bounding box in the format of (top left x, top left y, bottom right x, bottom right y).
        bbox2 (tuple): The second bounding box in the format of (top left x, top left y, bottom right x, bottom right y).
    
    Returns:
        List[float]: IoB intersection over bbox1 and bbox2.
    """
    if format not in ['xyxy', 'xywh']:
         raise ValueError('Invalid argument for format. Must be one of [xyxy, xywh]')
    
    if format == 'xywh':
        bb_test = xywh2xyxy(bb_test)
        bb_gt = xywh2xyxy(bb_gt)
    # print(bbox1, bbox2)
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate the area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of bboxes
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate and return the Intersection over Bbox (IoB)
    return max(intersection / area_bbox1, intersection / area_bbox2)


def plot_box(frame, bbox, color=(0,255,0)):
        """
        Plot a bounding box (xyxy) on a given frame.

        Args:
            frame (numpy.ndarray): The frame on which to plot the bounding box.
            bbox (tuple): The bounding box in the format of (top left x, top left y, bottom right x, bottom right y).
            color (tuple): The color of the bounding box in the format of (red, green, blue).

        Returns:
            numpy.ndarray: The frame with the bounding box plotted on it.
        """
        annotate_frame = frame.copy()
        annotator = Annotator(annotate_frame)
        for box in bbox:
            # transform the bbox to the root coordinate and plot
            annotator.box_label(box, color=color)
        annotate_frame = annotator.result()
        return annotate_frame