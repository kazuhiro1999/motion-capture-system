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

    def apply(self, image, crop_region=None, crop_size=None, threshold=15, return_mask=False):
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
        if return_mask:
            return np.where(mask==0, 0, 1)
        
        foreground = (image * mask[:,:,None]).astype(np.uint8)
        return foreground


import asyncio

def execute_async(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args, **kwargs)

from scipy.optimize import least_squares

class Tuner:
    def __init__(self, num_samples=5):
        self.num_samples = num_samples
        self.segmentation = Segmentation()
        self.reset()

    def reset(self):
        self.slot = None
        self.active = False
        self.isWaiting = False
        self.optimizing = False
        self.images = []
        self.crops = []
        self.seg_masks = []
        self.sub_masks = []
        self.process_end = False
        self.queue = []
        self.text = None
        self.score = 0

    def activate(self, slot, text):
        # sampling start
        self.slot = slot
        self.text = text
        self.text.update('Sampling Images ...')
        self.active = True
        return

    def process(self):
        if self.isWaiting:
            return 
        if len(self.queue) > 0:
            img = self.queue.pop(0)
            cv2.imshow(f'sample {len(self.seg_masks)}', img)
        if len(self.seg_masks) >= self.num_samples:
            if not self.optimizing:
                self.optimizing = True
                self.text.update('Optimizing Params ...')
                execute_async(self.optimize)
            else:
                return
        else:
            frame = self.slot.get_image()
            crop_region = self.slot.crop_region
            self.images.append(frame)
            self.crops.append(crop_region)
            img = crop_and_resize(frame, crop_region=crop_region, crop_size=[224,224]).astype(np.uint8)

            self.isWaiting = True
            execute_async(self.detect_person, img)

    def detect_person(self, image):
        img = image.copy()
        mask = self.segmentation.detect_person(img)
        self.seg_masks.append(mask)
        self.queue.append((image*mask[:,:,None]).astype(np.uint8))
        self.isWaiting = False

    def optimize(self):
        subtractor = self.slot.background_subtractor

        def func(a_min, a_max, c_min, c_max):
            subtractor.a_min = a_min
            subtractor.a_max = a_max
            subtractor.c_min = c_min
            subtractor.c_max = c_max

            loss = 0
            for image, crop_region, seg_mask in zip(self.images, self.crops, self.seg_masks):
                mask = subtractor.apply(image, crop_region, [224,224], return_mask=True)
                precision = np.count_nonzero(mask & seg_mask) / np.count_nonzero(mask)
                recall = np.count_nonzero(mask & seg_mask) / np.count_nonzero(seg_mask)
                f1 = 2*precision*recall/(precision+recall)
                loss += (1-f1)
            return loss / self.num_samples

        d = 0.05
        l_min = 100
        for _a_min in np.linspace(0.5,1,11):
            for _a_max in np.linspace(_a_min,1.5,int((1.5-_a_min)/d)+1):
                _c_min = 0
                _c_max = 30
                #for _c_min in np.linspace(0,12,4):
                    #for _c_max in np.linspace(_c_min,30,int((30-_a_min)/6)+1):
                loss = func(_a_min, _a_max, _c_min, _c_max)
                if loss < l_min:
                    l_min = loss
                    a_min = _a_min
                    a_max = _a_max
                    c_min = _c_min
                    c_max = _c_max
        
        subtractor.a_min = a_min
        subtractor.a_max = a_max
        subtractor.c_min = c_min
        subtractor.c_max = c_max  
        self.score = (1-loss)*100
        self.process_end = True
        return


import torch
import torchvision.transforms as T
   
class Segmentation:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
        self.model.eval()

    def detect_person(self, image):
        h,w = image.shape[:2]
        trf = T.Compose([
            T.ToPILImage(),
            T.Resize(800),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        inpt = trf(image).unsqueeze(0).to(self.device)
        out = self.model.to(self.device)(inpt)['out']
        mask = torch.argmax(out.squeeze(), dim=0).detach().byte().cpu().numpy()
        mask = cv2.resize(mask, (w,h))
        mask = np.where(mask==15,1,0)
        return mask


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