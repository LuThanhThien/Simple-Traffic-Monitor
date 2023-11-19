import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import imutils
from typing import Union, List
from datetime import datetime

default_save = 'output\preprocess'

def crop(frame, bbox, save=False):
    """
    Crop a frame based on a bounding box in pixel units.

    Args:
        frame (numpy.ndarray): The input frame to crop.
        bbox (list): A list of four integers representing the bounding box in pixel units (x, y, width, height).

    Returns:
        numpy.ndarray: The cropped frame.
    """
    x, y, w, h = bbox
    crop_frame = frame[y:y+h, x:x+w]
    if save:
        sufflix = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        save_display(img=crop_frame, project=default_save, name=f'crop-{sufflix}', show=False)
    return np.ascontiguousarray(crop_frame)


def display(image: Union[str, np.ndarray], dpi:int=100):
    """
    Display an image input as path or np.ndarray using matplotlib.
    """
    if type(image) == str:
        image = plt.imread(image)
    elif type(image) != np.ndarray:
        image = np.array(image)  
    elif type(image) == np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width  = image.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')
    ax.imshow(image, cmap='gray')

    plt.show()


def save_display(img=None, project=default_save, name='image', show=True):
    """
    Save and display image
    """
    os.makedirs(project, exist_ok=True)
    cv2.imwrite(f'{project}/{name}.jpg', img)
    print("Saved image to " + f'{project}/\033[1m{name}.jpg\033[0m')
    if show:    
        display(f'{project}/{name}.jpg')


def apply_method(method, img=None, project=default_save, name='image-preprocess', save=False, show=True, **kwargs):
    """
    Apply a method to an image and save, display the result.
    """
    new_img = method(img, **kwargs)
    if save:
        save_display(img=new_img, project=project, name=name, show=show)
    if show:
        display(new_img)
    return new_img


# simple image processing
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bitwise_not(img):
    return cv2.bitwise_not(img)

def hvs(img):
    img_hvs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hvs

def color_shift(img):
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_color

def blur(img):
    h, w = img.shape[:2]
    # print(h, w)
    if h + w > 1200:
        img_blur = cv2.GaussianBlur(img, (45, 45), 10)  
        return img_blur
    elif h + w > 800:
        img_blur = cv2.GaussianBlur(img, (33, 33), 10)
        return img_blur
    else:
        return False

def rotate(image, degree):
    return imutils.rotate_bound(image, degree)

def resize(img, min_w=500):
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    _, w, _ = img.shape
    scale = max(1, min_w / w)
    resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return resized_img

def threshold(img, thresh=170, type=cv2.THRESH_BINARY):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(img, thresh, 255, type)[1]

def noise_removal(img):
    return cv2.bilateralFilter(img, 11, 17, 17)

def smooth_font(img):
    image = cv2.bitwise_not(img)
    kernal = np.ones((3,3), np.uint8)
    image = cv2.erode(image, kernal, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def morph(img):
    kernel = np.ones((3,3), np.uint8)
    img_morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)
    return img_morph

def shadow_remove(img):
    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return norm_img

def filter_noise(img):
    # separate horizontal and vertical lines to filter out spots outside the rectangle
    kernel = np.ones((5,2), np.uint8)
    vert = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    vert = noise_removal(vert)
    kernel = np.ones((2,5), np.uint8)
    horiz = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    horiz = noise_removal(horiz)
    # combine
    rect = cv2.add(horiz,vert)
    
    return rect

def sharpen(img):
    A = 1
    filter = np.array([[-1,-1,-1], 
                        [-1,8+A,-1], 
                        [-1,-1,-1]])
    img_sharp = cv2.filter2D(img, -1, filter)
    return img_sharp

def edging(img):
    img_edging = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_edging = cv2.GaussianBlur(img_edging, (3,3), 0) 
    sobelxy = cv2.Sobel(src=img_edging, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) 
    return sobelxy



# plate perspectives
def sort_points(bbox):
    sorted_bbox = sorted(bbox, key=lambda x: x[1])
    pts_top = sorted(sorted_bbox[:2], key=lambda x: x[0])
    pts_bottom = sorted(sorted_bbox[2:], key=lambda x: x[0], reverse=True)
    new_bbox = []
    new_bbox.extend(pts_top)
    new_bbox.extend(pts_bottom)
    return list(new_bbox)

def get_box(img):
    # get largest contour
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = None
    for c in contours:
        area_thresh = 0
        area = cv2.contourArea(c)
        if area > area_thresh:
            area = area_thresh
            big_contour = c

    # get rotated rectangle from contour
    if big_contour is None:
        print('No contour found!')
        return None
        
    rot_rect = cv2.minAreaRect(big_contour)
    box = cv2.boxPoints(rot_rect)
    box = np.intp(box)

    # arrange points: (top-left, top-right, bottom-right, bottom-left)
    box = sort_points(box)
        
    return box

def rotating(img, show=True):
    box = get_box(img)
    pts1 = np.float32(box)
    w = np.linalg.norm(pts1[0] - pts1[1])
    h = np.linalg.norm(pts1[0] - pts1[3])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_rotated = cv2.warpPerspective(img, matrix, (int(w), int(h)))
    save_display(img=img_rotated, project=default_save, name='img_rotated',show=show)
    return img_rotated

# COMBINED
def thresholding(img, project=default_save, save=True, show=True, **kwargs):
    # resize - gray - unnoise - threshold - thin - morph
    img_resize = apply_method(resize, img, name='img_resize', save=False)
    img_gray = apply_method(grayscale, img_resize, name='img_grey', save=False)
    img_unnoise = apply_method(noise_removal, img_gray, name='img_unnoise', save=True)
    # img_bw = apply_method(bitwise_not, img_unnoise, 'img_bw', save=True)
    img_thresh_raw =  apply_method(threshold, img_unnoise, name='img_thresh_raw', save=False, **kwargs)
    img_thin = apply_method(smooth_font, img_thresh_raw, name='img_thin', save=False)
    img_morph = apply_method(morph, img_thin, save=False)
    img_thresh = apply_method(filter_noise, img_morph, project=project, name='img_thresh', save=save, show=show)
    
    return img_thresh

def pipeline(img_paths, project=default_save, name='image-preprocess', save=False, show=False, **kwargs):
    if type(img_paths[0]) == str:
        img = cv2.imread(img_paths)
    else:
        img = img_paths
    
    img_preprocess = apply_method(method=resize, img=img, save=False, show=show, **kwargs)
    img_preprocess = apply_method(method=sharpen, project=project, img=img_preprocess, 
                                      name=f'{name}', save=save, show=show)
    
    return img_preprocess

