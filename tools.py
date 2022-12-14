import time
import cv2
from detector import SKELETON

class Checkpoint:
    def __init__(self, keys=[]):
        self.keys = keys
        self.history = 60
        self.checkpoints = {}
        self.reset_time()

    def check(self, key):
        if key not in self.keys:
            self.keys.append(key)
        self.checkpoints.setdefault(key, [])
        t = time.time() - self.t
        self.checkpoints[key].append(t)
        if len(self.checkpoints[key]) > self.history:
            self.checkpoints[key].pop(0)

    def reset_time(self):
        self.t = time.time()

    def get_results(self, keys=None):
        if keys is None:
            keys = list(self.checkpoints.keys())
        results = {}
        for key in keys:
            array = self.checkpoints[key]
            mean_t = sum(array) / len(array)
            results[key] = f'{mean_t:.3f}'
        return results


class Clock:
    def __init__(self, fps):
        self.fps = fps


def draw_keypoints2d(image, keypoints2d, s=3, min_score=0.2):
    if keypoints2d is None:
        return image
    for x,y,c in keypoints2d:
        if c > min_score:
            cv2.circle(image, center=(int(x),int(y)), radius=s, color=(0,0,255))
    return image

def draw_skeleton(image, keypoints2d, min_score=0.2):
    if keypoints2d is None:
        return image
    for j,k in SKELETON:
        if keypoints2d[j,2] > min_score and keypoints2d[k,2] > min_score:
            x1,y1 = keypoints2d[j,:2]
            x2,y2 = keypoints2d[k,:2]
            cv2.line(image, (int(x1),int(y1)), (int(x2),int(y2)), color=(0,0,255))
    return image

def draw_crop_area(image, isdetected, crop_region):
    H,W = image.shape[:2]
    x1 = int(crop_region['x_min'] * W)
    y1 = int(crop_region['y_min'] * H)
    x2 = int(crop_region['x_max'] * W)
    y2 = int(crop_region['y_max'] * H)
    if isdetected:
        cv2.rectangle(image, (x1,y1), (x2,y2), color=(0,255,0), thickness=3)
    else:
        cv2.rectangle(image, (x1,y1), (x2,y2), color=(0,0,255), thickness=3)
    return image