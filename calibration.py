import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from window import BaseWindow
import PySimpleGUI as sg
from detector import KEYPOINT_DICT


class CameraSetting:
    def __init__(self, setting=None):
        self.camera_matrix = np.zeros([3,3])
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)
        self.proj_matrix = np.eye(3,4)
        self.load_setting(setting)

    def load_setting(self, setting):
        return 

    def set_camera_matrix(self, matrix=None, FOV=None, width=None, height=None):
        if matrix is not None:
            matrix = np.array(matrix)
        elif FOV is not None and width is not None and height is not None:
            focal = (width/2) / np.tan(np.radians(FOV/2))
            matrix = np.array([[focal,0,(width/2)],[0,focal,(height/2)],[0,0,1]])
        assert matrix.shape == (3,3)
        self.camera_matrix = matrix
        return self.camera_matrix

    def set_transform(self, pos=None, rot=None):
        if pos is not None:
            assert len(pos) == 3
            pos = np.array(pos).flatten()
            if pos is not None:
                self.position = pos
        if rot is not None:
            rot = np.array(rot)
            if rot.shape == (3,3):
                rot = cv2.Rodrigues(rot)[0]
            assert len(rot) == 3
            rot = np.array(rot).flatten()
            if rot is not None:
                self.rotation = rot
        self.reset_projection_matrix()

    def reset_projection_matrix(self):
        R = cv2.Rodrigues(self.rotation)[0]
        t = self.position.reshape([3,1])
        Rc = R.T
        tc = -R.T @ t
        self.proj_matrix = self.camera_matrix.dot(np.concatenate([Rc,tc], axis=1))
        return self.proj_matrix


class CalibrationWindow(BaseWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Calibration Window'
        self.image_size = (480, 480)
        self.slots = []
        self.base_i = 0
        self.pair_i = 1
        self.num_samples = 100
        self.interval = 1
        self.count = 0
        self.running = False
        self.keypoints2d_dict = {}

    def reset_keypoints2d_dict(self):
        self.keypoints2d_dict = {}
        for slot in self.slots:
            self.keypoints2d_dict[slot.name] = []

    def open(self, slots):
        if len(slots) < 2: # calibration need at least 2 cameras
            return 
        self.slots = slots
        # temp setting
        for slot in slots:
            slot.camera_setting.set_camera_matrix(matrix=None, FOV=90, width=640, height=360)
            slot.draw_image = True
            slot.subtract = True
            slot.pose_estimation = True
            slot.draw_crop_area = True
            slot.draw_keypoints2d = True

        self.reset_keypoints2d_dict()
        layout = [
            [sg.Text('Base Camera'), sg.Combo([slot.name for slot in slots], default_value=slots[self.base_i].name, enable_events=True, key='-SelectBase-')],
            [sg.Text('Pair Camera'), sg.Combo([slot.name for slot in slots], default_value=slots[self.pair_i].name, enable_events=True, key='-SelectPair-')],
            [sg.Graph(self.image_size, (0,0), self.image_size, key='-Graph-')],
            [sg.Button('Calibrate', enable_events=True, key='-Calibrate-'),sg.Text('n_samples'),sg.Input(default_text=str(self.num_samples), enable_events=True, key='N'),sg.Text('interval'),sg.Input(default_text=str(self.interval), enable_events=True, key='interval')],
            [sg.ProgressBar(max_value=self.num_samples, orientation='h', size=(20,20), key='-Bar-')]
        ]
        super().open(self.title, layout)
        
        self.draw_calibration_results([])

    def handle(self):
        event, values = self.window.read(timeout=0)
        if event == '-SelectBase-':
            name = values[event]
            self.base_i = [slot.name for slot in self.slots].index(name)
        if event == '-SelectPair-':
            name = values[event]
            self.pair_i = [slot.name for slot in self.slots].index(name)
        if event == 'N':
            try:
                self.num_samples = int(values[event])
                self.window['-Bar-'].update(max=self.num_samples)
            except:
                pass
        if event == 'interval':
            try:
                self.interval = int(values[event])
            except:
                pass
        if event == '-Calibrate-':
            # Sampling Start
            # プログレスバーやサンプル画像、人物検出を可視化する！
            self.running = True
            self.window['-Bar-'].update(0)
        if event is None:
            self.close()

        if not self.running:
            return 

        self.count += 1
        if self.count < self.interval:
            return
        self.count = 0
        n = len(self.keypoints2d_dict[self.slots[self.base_i].name])
        self.window['-Bar-'].update(n, max=self.num_samples)
        if  n >= self.num_samples: # サンプリング終了
            camera_setting_list = [slot.camera_setting for slot in self.slots]
            keypoints2d_list = np.array([self.keypoints2d_dict[slot.name] for slot in self.slots])
            keypoints3d_list = calibrate_cameras(camera_setting_list, keypoints2d_list, base_i=self.base_i, pair_i=self.pair_i)

            self.draw_calibration_results(keypoints3d_list)
            self.running = False
            self.reset_keypoints2d_dict()
        else:
            for slot in self.slots:
                if not slot.isDetected:
                    return
            for slot in self.slots:
                self.keypoints2d_dict[slot.name].append(slot.keypoints2d[:,:2])
        return

    def draw_calibration_results(self, keypoints3d_list):
        self.window['-Graph-'].erase()
        cx, cy = self.image_size[0]//2, self.image_size[1]//2
        self.window['-Graph-'].draw_line((0, cy), (cx*2, cy)) 
        self.window['-Graph-'].draw_line((cx, 0), (cx, cy*2)) 
        for slot in self.slots:
            print(slot.name)
            print(slot.camera_setting.position)
            print(slot.camera_setting.rotation)
            pos = slot.camera_setting.position
            rot = slot.camera_setting.rotation
            t = pos.reshape([3,1])
            R = cv2.Rodrigues(rot)[0]
            t_ = R @ np.array([0,0,1]) + t
            self.window['-Graph-'].draw_point((cx + pos[0]*50, cy + pos[2]*50), size=10) 
            #self.window['-Graph-'].draw_line((cx + pos[0]*50, cy + pos[2]*50), (cx + t_[0,0]*50, cy + t_[2,0]*50), width=1)
        return


def estimate_initial_extrinsic(pts1, pts2, K):
    # pts : (N,2)
    E, mask = cv2.findEssentialMat(pts2, pts1, K, method=cv2.FM_LMEDS)
    pts, R, t, mask = cv2.recoverPose(E, pts2, pts1, K)
    return R, t

def calibrate_cameras(camera_setting_list, keypoints2d_list, base_i=0, pair_i=1):
    n_views, n_frames, n_joints = keypoints2d_list.shape[:3]

    pts1 = keypoints2d_list[base_i].reshape([-1,2])
    pts2 = keypoints2d_list[pair_i].reshape([-1,2])
    K = camera_setting_list[base_i].camera_matrix
    R, t = estimate_initial_extrinsic(pts1, pts2, K)
    camera_setting_list[base_i].set_transform(np.zeros([3,1]), np.eye(3,3))
    camera_setting_list[pair_i].set_transform(t, R)

    points3d = cv2.triangulatePoints(
        camera_setting_list[base_i].proj_matrix,
        camera_setting_list[pair_i].proj_matrix,
        pts1.T,
        pts2.T
    )
    points3d = (points3d[:3,:] / points3d[3,:]).T

    keypoints3d_list = points3d.reshape([n_frames, n_joints, 3])

    for view_i in range(n_views):
        if view_i == base_i or view_i == pair_i:
            continue
        pts = keypoints2d_list[view_i].reshape([-1,2]).astype(np.float32)
        ret, rc, tc, mask = cv2.solvePnPRansac(points3d, pts, K, np.zeros([4,1]))
        Rc = cv2.Rodrigues(rc)[0]
        R = Rc.T
        t = -Rc.T @ tc
        camera_setting_list[view_i].set_transform(t, R)

    room_calibration(camera_setting_list, keypoints3d_list)

    return keypoints3d_list


def room_calibration(camera_setting_list, keypoints3d_list):
    o_pos = determine_center_position(keypoints3d_list)
    o_mat = determine_forward_rotation(keypoints3d_list)
    scale = determine_scale(keypoints3d_list, Height=1.6)
    o_rot = Rotation.from_matrix(o_mat)

    for camera_setting in camera_setting_list:
        t = o_rot.apply(camera_setting.position - o_pos).reshape([3,1]) * scale
        R = (o_rot * Rotation.from_euler('zyx', camera_setting.rotation)).as_matrix()
        camera_setting.set_transform(t, R)


def determine_center_position(keypoints3d_list):
    # keypoints3d_list : (n_frames, n_joints, 3)
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    center_position = m_foot.mean(axis=0)
    return center_position


def determine_forward_rotation(keypoints3d_list):
    # keypoints3d_list : (n_frames, n_joints, 3)
    l_shoulder = keypoints3d_list[:, KEYPOINT_DICT['left_shoulder']]
    r_shoulder = keypoints3d_list[:, KEYPOINT_DICT['right_shoulder']]
    l_hips = keypoints3d_list[:, KEYPOINT_DICT['left_hip']]
    r_hips = keypoints3d_list[:, KEYPOINT_DICT['right_hip']]
    m_hips = (l_hips + r_hips) / 2
    forward_vector = np.cross(l_shoulder - m_hips, r_shoulder - m_hips).mean(axis=0)
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    forward_global = np.array([0,0,1])
    rotation_matrix = (forward_vector.reshape([3,1]) @ forward_global.reshape([1,3])).astype(np.float32)
    return rotation_matrix

def determine_scale(keypoints3d_list, Height=1.0):
    head = keypoints3d_list[:, KEYPOINT_DICT['nose']]
    l_foot = keypoints3d_list[:, KEYPOINT_DICT['left_ankle']]
    r_foot = keypoints3d_list[:, KEYPOINT_DICT['right_ankle']]
    m_foot = (l_foot + r_foot) / 2
    height = np.linalg.norm(head - m_foot, axis=-1).mean()
    scale = Height / height
    return scale




