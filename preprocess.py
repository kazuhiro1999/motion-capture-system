import time
import numpy as np
import cupy as cp
import cv2

from detector import init_crop_region

class BackgroundSubtractor:
    def __init__(self, background=None, img_path='', a_min=0.7, a_max=1.0, c_min=0, c_max=30):
        self.background = background
        self.img_path = img_path
        self.a_min = a_min
        self.a_max = a_max
        self.c_min = c_min
        self.c_max = c_max

        if img_path != '':
            self.load_background(img_path)

    def load_config(self, config):
        self.a_min = config['a_min']
        self.a_max = config['a_max']
        self.c_min = config['c_min']
        self.c_max = config['c_max']
        if config['background_path'] != '':
            self.load_background(config['background_path'])
        

    def set_background(self, background):
        self.background = background

    def load_background(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.set_background(image)
        self.img_path = img_path

    def apply(self, image, crop_region=None, crop_size=None):
        if self.background is None:
            return image
        background = crop_and_resize(self.background, crop_region, crop_size)
        image = crop_and_resize(image, crop_region, crop_size)

        E = background.astype(np.int64)
        I = image.astype(np.int64)
        a = (E*I).sum(axis=-1) / np.maximum((E*E).sum(axis=-1), 1e-5)
        c = np.linalg.norm(I - a[:,:,None]*E, axis=-1)
        cond = (self.a_min <= a) & (a <= self.a_max) & (self.c_min <= c) & (c <= self.c_max)
        mask = np.where(cond, 0, 1)
        mask = cv2.medianBlur(mask.astype(np.float32), 5)
        mask = (mask * self.detect_diffs(image, background))[:,:,None]
        foreground = (image * mask).astype(np.uint8)

        return foreground

    def detect_diffs(self, image, background, threshold=15):
        img_gray = rgb_to_gray(image)
        bg_gray = rgb_to_gray(background)
        diffs = np.abs(img_gray - bg_gray)
        mask = np.where(diffs>threshold, 1, 0).astype(np.float32)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        return mask


class BackgroundSubtractorGPU:
    def __init__(self, background=None, img_path='', a_min=0.7, a_max=1.0, c_min=0, c_max=30):
        self.background = background
        self.img_path = img_path
        self.a_min = a_min
        self.a_max = a_max
        self.c_min = c_min
        self.c_max = c_max

        if img_path != '':
            self.load_background(img_path)

        cp.ones(2)*2

    def load_config(self, config):
        self.a_min = config['a_min']
        self.a_max = config['a_max']
        self.c_min = config['c_min']
        self.c_max = config['c_max']
        if config['background_path'] != '':
            self.load_background(config['background_path'])
        

    def set_background(self, background):
        self.background = background
        self.set_gpu(background.shape)

    def load_background(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.set_background(image)
        self.img_path = img_path

    def set_gpu(self, shape):
        self.image_gpu = cp.ndarray(shape, dtype=np.int64)
        self.background_gpu = cp.ndarray(shape, dtype=np.int64)

        cp.ones(2)*2

    def apply(self, image, crop_region=None, crop_size=None, threshold=15):
        if self.background is None:
            return image
        background = crop_and_resize(self.background, crop_region, crop_size)
        image = crop_and_resize(image, crop_region, crop_size)
        with cp.cuda.Device(0): 
            E_gpu = cp.asarray(background, dtype=cp.int64) 
            I_gpu = cp.asarray(image, dtype=cp.int64) 
            a_gpu = cp.divide(cp.sum(cp.multiply(E_gpu, I_gpu), axis=-1), cp.maximum(cp.sum(cp.multiply(E_gpu, E_gpu), axis=-1), 1e-5))
            c_gpu = cp.linalg.norm(I_gpu - cp.multiply(a_gpu[:,:,None], E_gpu), axis=-1)
            cond = (self.a_min <= a_gpu) & (a_gpu <= self.a_max) & (self.c_min <= c_gpu) & (c_gpu <= self.c_max)
            shadow_mask_gpu = cp.where(cond, 0, 1)

            img_gray = 0.299*I_gpu[:,:,0] + 0.587*I_gpu[:,:,1] + 0.114*I_gpu[:,:,2]
            bg_gray = 0.299*E_gpu[:,:,0] + 0.587*E_gpu[:,:,1] + 0.114*E_gpu[:,:,2]
            diffs = cp.abs(img_gray - bg_gray)
            mask_gpu = cp.where(diffs>threshold, 1, 0).astype(cp.float32)
            mask_gpu = shadow_mask_gpu * mask_gpu

        mask_cpu = cp.asnumpy(mask_gpu)
        mask = cv2.medianBlur(mask_cpu.astype(np.float32), 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
        foreground = (image * mask[:,:,None]).astype(np.uint8)

        return foreground

   

def rgb_to_gray(image):
    assert (image.ndim == 3 or image.ndim == 4)
    if image.ndim == 4:
        gray = 0.299*image[:,:,:,0] + 0.587*image[:,:,:,1] + 0.114*image[:,:,:,2]
        return gray[:,:,:,None] 
    elif image.ndim == 3:
        gray = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
        return gray[:,:,None] 
    else:
        return None

def crop_and_resize(image, crop_region, crop_size):
    if image.ndim == 2:
        image = image[:,:,None]
    image_height, image_width, channels = image.shape
    if crop_region is not None:
        if crop_region['width'] < 0.1 or crop_region['height'] < 0.1:
            crop_region = init_crop_region(image_height, image_width)
        x = int(crop_region['x_min']*image_width)
        y = int(crop_region['y_min']*image_height)
        w = int(crop_region['width']*image_width)
        h = int(crop_region['height']*image_height)
        image = image[y:y+h, x:x+w, :]
    if crop_size is not None:
        image = cv2.resize(image, dsize=crop_size).astype(np.float32)
    return image
    

def padding(cropped_image, crop_region, image_shape):
    W,H = image_shape
    img = np.zeros([H,W,3])
    x = int(crop_region['x_min']*W)
    y = int(crop_region['y_min']*H)
    w = int(crop_region['width']*W)
    h = int(crop_region['height']*H)
    img_cropped = cv2.resize(cropped_image, dsize=(w,h))
    img[y:y+h, x:x+w, :] = img_cropped
    return img

def to_pinned_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src