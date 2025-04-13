import cv2
import numpy as np
import os
from PIL import Image

def channel_wise_hist_eq(image):
    """Apply histogram equalization to each channel separately"""
    channels = cv2.split(image)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    return cv2.merge(eq_channels)

def hsv_hist_eq(image):
    """Convert to HSV space and equalize only Value channel"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

def clahe_hist_eq(image):
    """Apply CLAHE to L-channel in LAB color space"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def save_image(image, path, filename):
    """Save image to specified path"""
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    cv2.imwrite(full_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return full_path

def load_image(uploaded_file):
    """Load image from uploaded file"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)