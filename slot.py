import cv2
from preprocess import BackgroundSubtractor, BackgroundSubtractorGPU
from detector import init_crop_region

class Slot:
    def __init__(self, name, config=None):
        self.name = name
        self.mode = 'camera'
        self.camera_id = None
        self.video_path = ''
        self.cap = None
        self.isActive = False
        self.isPlaying = False
        self.frame = None
        self.draw_image = False
        self.background_subtractor = BackgroundSubtractorGPU()
        self.subtract = False
        self.pose_estimation = False
        self.crop_region = init_crop_region(1080,1920)
        self.draw_keypoints2d = False

        if config is not None:
            self.load_config(config)

    def load_config(self, config):
        if self.name in config.keys():
            self.video_path = config[self.name]['video_path']
            if self.video_path != '':
                self.set_video(self.video_path)
            self.background_subtractor.load_config(config[self.name])

    def set_video(self, video_path):
        if video_path == None:
            self.video_path = ''
            self.cap = None
            self.isActive = False
            self.frame = None
            return True

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.video_path = video_path
            self.cap = cap
            self.isActive = True
            return True
        else:
            return False

    def set_camera(self, camera_id):
        if camera_id == None:
            self.camera_id = None
            self.cap = None
            self.isActive = False
            self.frame = None
            return True

        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            self.camera_id = camera_id
            self.cap = cap
            self.isActive = True
            return True
        else:
            return False

    def read(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                return (True, frame)
        return (False, None)

    def get_image(self):
        return self.frame

    def open_setting(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        
    def get_config(self):
        cfg = {
            'name' : self.name, 
            'video_path' : self.video_path,
            'background_path' : self.background_subtractor.img_path,
            'a_min' : self.background_subtractor.a_min,
            'a_max' : self.background_subtractor.a_max,
            'c_min' : self.background_subtractor.c_min,
            'c_max' : self.background_subtractor.c_max
        }
        return cfg

    def close(self):
        if self.cap is not None:
            self.cap.release()