import time
import cv2

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


def draw_keypoints2d(image, keypoints2d, min_score=0.2):
    for x,y,c in keypoints2d:
        if c > min_score:
            cv2.circle(image, (x,y), radius=3, color=(0,0,255))
    return image