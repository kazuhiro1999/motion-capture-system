import time
import cv2
import PySimpleGUI as sg
from slot import Slot

class MainWindow:
    def __init__(self, slots=[]):
        self.window = None
        self.slots = slots

    def open(self):
        layout = [[self.get_frame_layout(slot.name) for slot in self.slots]]
        self.window = sg.Window('Main', layout=layout, finalize=True)

    def get_frame_layout(self, name):
        layout = [
            [sg.Button('カメラを選択', enable_events=True, key=f'{name}-Camera'),
             sg.Button('動画を開く', enable_events=True, key=f'{name}-Video')],
            [sg.Button('▶', enable_events=True, key=f'{name}-Play:Pause'),
             sg.Slider(range=(0,0), size=(20,5), orientation='h', enable_events=True, key=f'{name}-Slider')],
            [sg.Checkbox('画像を表示する', enable_events=True, key=f'{name}-Image')],
            [sg.Button('背景の設定', enable_events=True, key=f'{name}-Setting')],
            [sg.Checkbox('背景差分を適用する', enable_events=True, key=f'{name}-Subtraction')],
            [sg.Checkbox('クロップ領域を描画する', enable_events=True, key=f'{name}-CropArea')],
            [sg.Checkbox('姿勢推定を適用する', enable_events=True, key=f'{name}-PoseEstimation')],
            [sg.Checkbox('キーポイントを描画する', enable_events=True, key=f'{name}-Keypoints')],
            [sg.Button('パフォーマンスグラフを表示する', enable_events=True, key=f'{name}-Performance')],
        ]
        return sg.Frame(name, layout=layout, key=name)

    def close(self):
        self.window.close()
        self.window = None
    
    def read(self):
        return self.window.read(timeout=0)

    def handle_events(self, event):
        # return : slot, action
        if event is None:
            return None, 'Close'
        elif event == '__TIMEOUT__':
            return None, None
        else:
            slot_name, action = event.split('-')
            return self.get_slot(slot_name), action

    def get_slot(self, name):
        for slot in self.slots:
            if slot.name == name:
                return slot
        else:
            return None


class CameraWindow:
    def __init__(self):
        self.update_cameras()
        self.image_size = (640,480)
        self.window = None
        self.isActive = False
        self.slot = None
        
    # 使用できるカメラの更新
    def update_cameras(self):
        self.camera_id_list = []
        for i in range(10):
            cap = cv2.VideoCapture(i+cv2.CAP_DSHOW)
            if cap.isOpened():
                self.camera_id_list.append(i)
            cap.release()

        return self.camera_id_list

    def get_available_cameras(self):
        camera_list = ['None']
        for i in self.camera_id_list:
            camera_list.append(f'Camera {i}')
        return camera_list

    def get_default_value(self, slot):
        if slot.camera_id is None:
            return 'None'
        else:
            return f'Camera {slot.camera_id}'

    def open(self, slot):
        self.slot = slot
        self.camera_list = self.get_available_cameras()
        layout = [
            [sg.Graph(self.image_size, (0,0), self.image_size, key='-Graph-')],
            [sg.Combo(self.camera_list, default_value=self.get_default_value(slot), enable_events=True, key='-Select-')],
            [sg.Button('カメラ設定', enable_events=True, key='-Setting-'), sg.Button('OK', enable_events=True, key='-OK-')]
            ]
        self.window = sg.Window('カメラ選択', layout=layout, finalize=True)
        self.isActive = True

    def read(self):
        return self.window.read(timeout=0)

    def close(self):
        if self.window is not None:
            self.window.close()
        self.isActive = False
        self.slot = None

    def draw_image(self):
        if self.slot is None:
            return 
        if self.slot.isActive:
            image = self.slot.get_image()
            image = cv2.resize(image, dsize=self.image_size)
            img_bytes = cv2.imencode('.png', image)[1].tobytes()
            self.window['-Graph-'].erase()
            self.window['-Graph-'].draw_image(data=img_bytes, location=(0,self.image_size[1]))

    def handle(self):
        self.draw_image()
        event, values = self.window.read(timeout=0)
        if event == '-Select-':
            if values[event] == 'None':
                self.slot.set_camera(None)
            else:
                camera_id = int(values[event].split(' ')[1])
                self.slot.set_camera(camera_id)
        if event == '-Setting-':
            self.slot.open_setting()
        if event == '-OK-':
            self.close()
        if event is None:
            self.close()


class SettingWindow:
    def __init__(self):
        self.image_size = (640,480)
        self.window = None
        self.slot = None
        self.isActive = False
        self.self_timer = ['なし', '1 sec', '3 sec', '5 sec', '10 sec']
        self.wait_time = 0
        self.start_time = 0
        self.shot = False

    def open(self, slot):
        self.slot = slot
        layout = [
            [sg.Graph(self.image_size, (0,0), self.image_size, key='-Graph-')],
            [sg.Button('ファイルから選択', enable_events=True, key='-Select-')],
            [sg.Button('背景画像を撮影する', enable_events=True, key='-Shot-'), sg.Combo(self.self_timer, default_value='なし', enable_events=True, key='-Timer-')],
            [sg.Slider(range=(0,2), default_value=slot.background_subtractor.a_min, resolution=0.01, size=(20,5), orientation='h', enable_events=True, key='-a_min-')],
            [sg.Slider(range=(0,2), default_value=slot.background_subtractor.a_max, resolution=0.01, size=(20,5), orientation='h', enable_events=True, key='-a_max-')],
            [sg.Slider(range=(0,255), default_value=slot.background_subtractor.c_min, resolution=1, size=(20,5), orientation='h', enable_events=True, key='-c_min-')],
            [sg.Slider(range=(0,255), default_value=slot.background_subtractor.c_max, resolution=1, size=(20,5), orientation='h', enable_events=True, key='-c_max-')],
            [sg.Button('背景画像を確認', enable_events=True, key='-Show-'), sg.Button('背景画像を保存', enable_events=True, key='-Save-')],
            [sg.Button('OK', enable_events=True, key='-OK-')]
            ]
        self.window = sg.Window('背景設定', layout=layout, finalize=True)
        self.isActive = True
        if self.slot.background_subtractor.background is not None:
            self.show_background_image()

    def handle(self):
        self.draw_image()
        event, values = self.window.read(timeout=0)
        if event == '-Timer-':
            if values[event] == 'なし':
                self.wait_time = 0.0
            else:
                self.wait_time = float(values[event].split(' ')[0])
        if event == '-Shot-':
            self.shot = True
            self.start_time = time.time()
        if event == '-Select-':
            img_path = sg.popup_get_file('背景画像を選択してください')
            if img_path is not None:
                try:
                    img = cv2.imread(img_path)
                    self.slot.background_subtractor.set_background(img)
                    self.show_background_image()
                except:
                    sg.popup_ok(f'cannot open file : {img_path}')
        if event == '-Show-':
            self.show_background_image()
        if event == '-Save-':
            image = self.slot.background_subtractor.background
            if image is not None:
                path = sg.popup_get_file('保存先のファイルを選択', save_as=True)
                if path is not None:
                    cv2.imwrite(path, image)
            else:
                sg.popup_ok('背景画像がありません')
        if event == '-a_min-':
            self.slot.background_subtractor.a_min = values[event]
        if event == '-a_max-':
            self.slot.background_subtractor.a_max = values[event]
        if event == '-c_min-':
            self.slot.background_subtractor.c_min = values[event]
        if event == '-c_max-':
            self.slot.background_subtractor.c_max = values[event]
        if event == '-OK-':
            self.close()
        if event is None:
            self.close()
            
        if self.slot is None or not self.slot.isActive:
            self.shot = False
        
        if self.shot:
            if time.time() >= self.start_time + self.wait_time:
                image = self.slot.get_image()
                self.slot.background_subtractor.set_background(image)
                self.show_background_image()
                self.shot = False
       
    def draw_image(self):
        if self.slot is None:
            return 
        if self.slot.isActive:
            image = self.slot.get_image()
            image = self.slot.background_subtractor.apply(image)
            image = cv2.resize(image, dsize=self.image_size)
            img_bytes = cv2.imencode('.png', image)[1].tobytes()
            self.window['-Graph-'].erase()
            self.window['-Graph-'].draw_image(data=img_bytes, location=(0,self.image_size[1]))

    def show_background_image(self):
        if self.slot is not None and self.slot.background_subtractor.background is not None:
            cv2.imshow(f'{self.slot.name}-Background', self.slot.background_subtractor.background)
        else:
            sg.popup_ok('背景画像がありません')

    def close(self):
        if self.window is not None:
            self.window.close()
        self.isActive = False
        self.slot = None


class BaseWindow:
    def __init__(self):
        self.window = None
        self.isActive = False

    def open(self, title, layout):
        self.window = sg.Window(title, layout=layout, finalize=True)
        self.isActive = True

    def close(self):
        if self.window is not None:
            self.window.close()
        self.window = None
        self.isActive = False

class PerformanceWindow(BaseWindow):
    def __init__(self):
        super().__init__()

    def open(self, keys):
        title = f'Performance Window'
        layout = [[sg.Text(key), sg.Text('0.0', key=key)] for key in keys]
        super().open(title, layout)

    def update(self, results):
        for key, value in results.items():
            if key in self.window.key_dict.keys():
                self.window[key].update(value)