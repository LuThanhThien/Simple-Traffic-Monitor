import cv2
import numpy as np
import easyocr
import time
from src.util.utils import *
from src.util.preprocess import *
from src.config import *

default_save = 'reader\ocr'
ocr_classes = list(DICT_OCR_CLASSES.values())
ocr_indexes = list(DICT_OCR_CLASSES.keys())


# plot OCR result with confidence
def plot_ocr(image, result, color=(0, 165, 255)):
    bbox, text, conf = result
    label = f'[{text}] {int(conf*100)}%'
    cv2.drawContours(image, np.array([bbox]), -1, color, 2)
    overlay = image.copy()
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
    x0, y0 = (bbox[0][0], max(bbox[0][1]-10, 50))
    cv2.rectangle(overlay, (x0, y0+14), (x0+w, y0-h-14), color, -1)
    alpha = 0.4  # Transparency factor.
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.putText(image, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return image

# =====================TRANSFORM COORDINATES=====================================
def normalize_coordinates(point, img_w, img_h):
    x, y = point
    x /= img_w
    y /= img_h
    return x, y

def transform_coordinates(img, bbox, ocrbox):
    resize_h, resize_w = img.shape[:2]
    x, y, real_w, real_h = bbox
    x, y, real_w, real_h = int(x), int(y), int(real_w), int(real_h)
    transform_box = []
    for pt in ocrbox:
        x0, y0 = pt
        x0, y0 = normalize_coordinates((x0, y0), resize_w, resize_h)
        x0, y0 = int(x0 * real_w), int(y0 * real_h)
        transform_box.append((x0+x, y0+y))
    return transform_box
# =================================================================================


# =====================MERGE RESULTS=============================================
def boxes_packing(merge_boxes):
    points = np.array(merge_boxes).reshape(-1, 2)
    x1 = np.sort(points, axis=0)[0][0]
    x2 = np.sort(points, axis=1)[0][1]
    x3 = np.sort(points, axis=0)[-1][0]
    x4 = np.sort(points, axis=1)[-1][1]
    return [(x1, x2), (x3, x2), (x3, x4), (x1, x4)]


def merge_results(results):
    # this function used to merge results from multiple ocr boxes of one plate
    merge_text = ''
    average_conf = 0
    merge_boxes = []
    # merge text and conf
    for r in results:
        bbox, text, conf = r
        merge_text += text
        average_conf += conf
        merge_boxes.append(bbox)
    # average conf
    average_conf /= len(results)
    # merge boxes
    packed_box = boxes_packing(merge_boxes)

    return packed_box, merge_text, average_conf
# =================================================================================


def correct_ocr(ocr_detect, ocr_classes=ocr_classes):
    ocr_detect_correct = ''
    lp_classes = ocr_classes + ['-']
    for ocr in ocr_detect:
        if ocr.upper() in lp_classes:
            ocr_detect_correct += ocr.upper()
    return ocr_detect_correct


def read_ocr(result, ocr_classes=ocr_classes, verbose=False):
    correct_results = []
    for r in result:
        box, text, conf = r
        correct_text = correct_ocr(text, ocr_classes)
        if correct_text == '':
            continue
        ocrbox = [tuple(map(int, pt)) for pt in box]
        correct_results.append([ocrbox, correct_text, conf])
        if verbose:
            print('[{}] {}%'.format(correct_text, int(conf*100)))
    return correct_results



def easyocr_predict(imgs, bbox, reader, ocr_classes=ocr_classes, project=default_save,
                     name='image-easyocr', verbose=False, save=False, show=True):
    os.makedirs(project, exist_ok=True)
    print('[Predicting by EasyOCR]')
    eocr_results = reader.readtext_batched(imgs)
    results = []
    # loop through all images
    for i, img_result in enumerate(eocr_results):
        img = imgs[i]
        # read and plot for image index i
        correct_results = read_ocr(img_result, ocr_classes, verbose=verbose)
        # merge boxes and text if any
        final_result = merge_results(correct_results)
        # invert to frame coordinates
        packed_box, merge_text, average_conf = final_result
        ocrbox = transform_coordinates(img, bbox, packed_box)
        # append result
        results.append([ocrbox, merge_text, average_conf])
        # plot and save
        img = plot_ocr(img, final_result)
        if show and save:
            save_display(img=img, project=project ,name=f'{name}', show=show)
        if show and not save:
            display(im_data=img)
    return results



# =====================YOLO OCR==================================================
def yolo_results_extract(results, classes):
    yolo_results = []
    print(results)
    return

        

    return yolo_results

def yoloocr_predict(imgs, bbox, yolo_reader, ocr_classes=ocr_indexes, project=default_save,
                     name='image-yoloocr', verbose=False, show=False):
    os.makedirs(project, exist_ok=True)
    print('Predicting by YOLO-OCR >>>')
    yocr_results = yolo_reader.predict(imgs, conf=0.3, classes=ocr_classes, save=True, project=f'{project}\{name}')
    print(len(yocr_results))
    print(yocr_results[0].boxes)
    # results = []
    # for i, img_result in enumerate(yocr_results):
    #     img = imgs[i]
    #     # read and plot for image index i
    #     correct_results = read_ocr(img_result, ocr_classes, verbose=verbose)
    #     # merge boxes and text if any
    #     final_result = merge_results(correct_results)
    #     # invert to frame coordinates
    #     packed_box, merge_text, average_conf = final_result
    #     ocrbox = transform_coordinates(img, bbox, packed_box)
    #     print(ocrbox)
    #     # append result
    #     results.append([ocrbox, merge_text, average_conf])
    #     # plot and save
    #     img = plot_ocr(img, final_result)
    #     save_display(img=img, project=project ,name=f'{name}', show=show)
    # return results
