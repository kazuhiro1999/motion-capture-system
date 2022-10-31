import cv2
import numpy as np
import itertools


class Triangulator:
    def __init__(self, slots, num_keypoints=17, weights=None, min_score=0.2):
        self.slots = slots
        self.n_views = len(slots)
        self.num_keypoints = num_keypoints
        self.weights = weights
        self.min_score = min_score
        self.isActive = False

    def triangulate(self):
        isActive = [slot.isActive and slot.pose_estimation and (slot.proj_matrix is not None) for slot in self.slots]
        active_slots = [slot for slot in self.slots if slot.isActive and slot.pose_estimation and (slot.proj_matrix is not None)]
        # 2D姿勢推定が有効なスロットが2以下の場合は3D推定不可
        if len(active_slots) < 2:
            return None
        keypoints3d = []
        for k in range(self.num_keypoints):
            isDetected = [slot.keypoints2d[k,2]>self.min_score for slot in active_slots]
            if isDetected.count(True) < 2:
                # 3D推定失敗
                keypoints3d.append(np.array([0,0,0]))
            else:
                points3d = []
                slots = [slot for slot, detected in zip(active_slots, isDetected) if detected]
                for slot1, slot2 in itertools.combinations(slots, 2):
                    point3d = cv2.triangulatePoints(
                        slot1.proj_matrix,
                        slot2.proj_matrix,
                        slot1.keypoints2d[k,:2].T,
                        slot2.keypoints2d[k,:2].T
                    )
                    point3d = (point3d[:3,:] / point3d[3,:]).flatten()
                    points3d.append(point3d)
                if self.weights is None:
                    keypoint3d = np.array(points3d).mean(axis=0)
                keypoints3d.append(keypoint3d)

        return np.array(keypoints3d)


def get_proj_matrix(rc, tc, K):
    rc = np.array(rc).reshape([3,1])
    Rc, _ = cv2.Rodrigues(rc)
    tc = np.array(tc).reshape([3,1])
    K = np.array(K)
    proj_matrix = K.dot(np.concatenate([Rc,tc], axis=1))
    return proj_matrix



class NoiseRemover:
    def __init__(self, history, weights=None):
        if weights is not None:
            assert len(weights) == history
        self.history = history
        self.weights = weights
        self.series = []
        self.isActive = False

    def apply(self, data):
        if data is not None:
            self.series.append(data)
        else:
            pass # 時系列予測で補間する
        if len(self.series) > self.history:
            self.series.pop(0)
        elif len(self.series) < 1:
            return None

        series = np.array(self.series)
        if self.weights is not None and len(series)==self.history:
            return (np.array(self.weights) * series).sum(axis=0)
        else:
            return series.mean(axis=0)

    


