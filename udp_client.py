import numpy as np
import json
import socket
import cv2

class UDPClient:
    def __init__(self, host, port, buffersize=1024):
        self.host = host
        self.port = port
        self.buffersize = buffersize
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host,port))
        print('listening...')

    def listen(self):
        data, addr = self.sock.recvfrom(4096)
        json_data = json.loads(data)
        return json_data
    
    def close(self):
        self.sock.close()

def to_keypoints3d(data):
    keypoints3d = []
    for label in list(data.keys()):
        x = data[label]['x']
        y = data[label]['y']
        z = data[label]['z']
        keypoints3d.append(np.array([x,y,z]))
    return np.array(keypoints3d)

def draw_keypoints3d(image, keypoints3d, proj_matrix):
    points3d = np.concatenate([keypoints3d.T, np.ones([1,17])], axis=0)
    keypoints2d = proj_matrix @ points3d
    keypoints2d = (keypoints2d[:2,:] / keypoints2d[2,:]).T
    for x,y in keypoints2d:
        cv2.circle(image, (int(x),int(y)), radius=10, color=(255,255,255))
    return image


if __name__ == '__main__':

    with open('camera_settings.json', 'r') as f:
        camera_settings = json.load(f)

    proj_matrix = np.array(camera_settings['setting1_c01']['proj_matrix'])

    client = UDPClient(host='', port=2435, buffersize=4096)

    while True:
        data = client.listen()
        keypoints3d = to_keypoints3d(data)
        print(keypoints3d[6])
        image = np.zeros([360,640,3])
        image = draw_keypoints3d(image, keypoints3d, proj_matrix)
        #image = cv2.resize(image, dsize=[480,270])
        cv2.imshow('keypoints3d', image)
        if cv2.waitKey(1) == ord('s'):
            break

    client.close()


