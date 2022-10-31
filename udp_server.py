import socket
import json


class UDPServer:
    def __init__(self, host='127.0.0.1', port='4444'):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isActive = False

    def send(self, data):
        try:
            data_json = json.dumps(data)
            self.client.sendto(data_json.encode('utf-8'), (self.host, self.port))
            return True
        except:
            return False

    def close(self):
        self.client.close()


KEYPOINT_DICT = {
  'nose':0,
  'left_eye':1,
  'right_eye':2,
  'left_ear':3,
  'right_ear':4,
  'left_shoulder':5,
  'right_shoulder':6,
  'left_elbow':7,
  'right_elbow':8,
  'left_wrist':9,
  'right_wrist':10,
  'left_hip':11,
  'right_hip':12,
  'left_knee':13,
  'right_knee':14,
  'left_ankle':15,
  'right_ankle':16
}

def keypoints3d_to_data(keypoints3d):
    data = {}
    if keypoints3d is not None:
        for k, label in enumerate(KEYPOINT_DICT.keys()):
            data[label] = {'x':keypoints3d[k,0], 'y':keypoints3d[k,1], 'z':keypoints3d[k,2] }
    return data