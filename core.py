import onnxruntime
import numpy as np
import cv2
from preprocess import crop_and_resize, rgb_to_gray
from detector import detect_keypoints, transform_keypoints


def run_interface(session, image):
    input_image = rgb_to_gray(image) / 255
    inputs = input_image.astype(np.float32)[None,:,:,:]
    outputs = session.execute(inputs)
    keypoints2d = detect_keypoints(outputs[0])
    return keypoints2d

class Session:
    def __init__(self, model_path, executeType='cpu'):
        assert executeType in ['cpu', 'gpu']
        try:
            if executeType == 'cpu':
                self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            elif executeType == 'gpu':
                self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            self.inputs = [i.name for i in self.session.get_inputs()]
            self.input_name = self.session.get_inputs()[0].name
            self.outputs = [output.name for output in self.session.get_outputs()]
            print(f'model successfully loaded : {model_path}')
        except:
            print(f'model cannot be loaded : {model_path}')

    def execute(self, x):
        inputs = {self.input_name : x}
        outputs = self.session.run(self.outputs, inputs)
        return outputs


if __name__ == '__main__':
    session = Session('models/pose2d_mobile_gray_random_100.onnx', 'cpu')
