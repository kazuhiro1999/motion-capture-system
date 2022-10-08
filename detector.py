'''
人物領域の検出
MoveNetのチュートリアルを参照

'''
import numpy as np
import cv2

MIN_CROP_KEYPOINT_SCORE = 0.2


# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def detect_keypoints(heatmaps, method='average'):
    N,H,W,K = heatmaps.shape
    i = np.arange(56)
    j = np.arange(56)
    coord_y, coord_x = np.meshgrid(i,j,indexing='ij')
    x,y,confidence = [],[],[]
    for heatmap in heatmaps:
        for k in range(K):
            c = heatmap[:,:,k].max()
            if c < 0.2:
                x.append(0)
                y.append(0)
                confidence.append(0)
                continue
            x.append(np.average(coord_x, weights=heatmap[:,:,k]))
            y.append(np.average(coord_y, weights=heatmap[:,:,k]))
            confidence.append(c)
    x = np.array(x).reshape([N,1,K]) * 224 / 56
    y = np.array(y).reshape([N,1,K]) * 224 / 56
    confidence = np.array(confidence).reshape([N,1,K])
    keypoints2d = np.stack([x, y, confidence], axis=3)
    return keypoints2d

# キーポイントをクロップ画像に合わせて変換
def transform_keypoints(keypoints_with_scores, image_height, image_width, crop_region, crop_size):
    p = [image_width * crop_region['x_min'], image_height * crop_region['y_min']]
    rh = (image_height * crop_region['height']) / crop_size[0]
    rw =(image_width * crop_region['width']) / crop_size[1]
    keypoints = keypoints_with_scores[0,0,:,:].copy()
    keypoints[:,0] = keypoints[:,0] * rw
    keypoints[:,1] = keypoints[:,1] * rh
    keypoints[:,:2] = keypoints[:,:2] + p
    return keypoints


# 人物が検出できなかった場合は中央をできるだけ大きくクロップ
def init_crop_region(image_height, image_width):
    if image_width < image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }

# 人物が検出されているか
def torso_visible(keypoints):
    return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] > MIN_CROP_KEYPOINT_SCORE or 
             keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] > MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] > MIN_CROP_KEYPOINT_SCORE))


def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0]);
        dist_x = abs(center_x - target_keypoints[joint][1]);
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y

        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


# 人物領域の検出
def determine_crop_region(keypoints, image_height, image_width):
    keypoints = keypoints.copy().reshape([1,1,17,3])
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [keypoints[0, 0, KEYPOINT_DICT[joint], 0],
                                   keypoints[0, 0, KEYPOINT_DICT[joint], 1]]

    if torso_visible(keypoints):
        center_y = (target_keypoints['left_hip'][0] + target_keypoints['right_hip'][0]) / 2;
        center_x = (target_keypoints['left_hip'][1] + target_keypoints['right_hip'][1]) / 2;

        (max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange) = determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x)
        
        crop_length_half = np.amax([max_torso_xrange * 2.2, max_torso_yrange * 2.2,
                                    max_body_yrange * 1.2, max_body_xrange * 1.2])
        
        tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)]);
        
        crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2;
            return {
                'y_min': crop_corner[1] / image_height,
                'x_min': crop_corner[0] / image_width,
                'y_max': (crop_corner[1] + crop_length) / image_height,
                'x_max': (crop_corner[0] + crop_length) / image_width,
                'height': (crop_corner[1] + crop_length) / image_height - crop_corner[1] / image_height,
                'width': (crop_corner[0] + crop_length) / image_width - crop_corner[0] / image_width
            }
    else:
        return init_crop_region(image_height, image_width)

   
# クロップ領域を画面最大で横移動
def adjust_crop_region(keypoints, image_height, image_width):

    bbox_length = min(image_width, image_height)
    c = keypoints[:,0].mean()
    box_height = bbox_length / image_height
    box_width = bbox_length / image_width
    y_max = 1.0
    y_min = 0.0
    x_max = (c + (bbox_length / 2)) / 960
    x_min = (c - (bbox_length / 2)) / 960
    if x_max > 1.0:
        x_min -= (x_max-1)
        x_max = 1.0
    if x_min < 0.0:
        x_max += (-x_min)
        x_min = 0.0
    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_max,
        'x_max': x_max,
        'height': box_height,
        'width': box_width
    }