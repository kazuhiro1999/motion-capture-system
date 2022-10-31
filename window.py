import time
import cv2
import PySimpleGUI as sg
from slot import Slot
import numpy as np
import json

class MainWindow:
    def __init__(self, slots=[]):
        self.window = None
        self.slots = slots
        self.config = None

    def open(self, config=None):
        layout = [
            [self.get_frame_layout(slot.name) for slot in self.slots],
            [sg.Button('▶', enable_events=True, key=f'Main-Play:Pause')],
            [sg.Checkbox('3次元復元を適用する', enable_events=True, key='Main-Triangulation'),
             sg.Combo(['average', 'weighted'], default_value='average', enable_events=True, key='Main-TriangulateMethod')],
            [sg.Checkbox('時系列平滑化を適用する', enable_events=True, key='Main-Smoothing')],
            [sg.Checkbox('UDP送信を有効化', enable_events=True, key='Main-UDP')]
        ]
        self.window = sg.Window('コントロールパネル', layout=layout, finalize=True)

        if config is not None:
            self.load_config(config)

    def get_frame_layout(self, name):
        layout = [
            [sg.Button('カメラを選択', enable_events=True, key=f'{name}-Camera'),
             sg.Button('動画を開く', enable_events=True, key=f'{name}-Video')],
            [sg.Button('▶', enable_events=True, key=f'{name}-Play:Pause'),
             sg.Slider(range=(0,0), size=(20,5), orientation='h', enable_events=True, key=f'{name}-Slider')],
            [sg.Checkbox('画像を表示する', default=True, enable_events=True, key=f'{name}-Image'),
             sg.Combo(['Auto', '640x360','640x480','320x180'], default_value='Auto', enable_events=True, key=f'{name}-Size')],
            [sg.Button('背景の設定', enable_events=True, key=f'{name}-Setting')],
            [sg.Checkbox('背景差分を適用する', enable_events=True, key=f'{name}-Subtraction')],
            [sg.Checkbox('クロップ領域を描画する', enable_events=True, key=f'{name}-CropArea')],
            [sg.Checkbox('姿勢推定を適用する', enable_events=True, key=f'{name}-PoseEstimation')],
            [sg.Checkbox('キーポイントを描画する', enable_events=True, key=f'{name}-Keypoints')],
            [sg.Button('カメラパラメータの設定', enable_events=True, key=f'{name}-ProjMatrix')],
            [sg.Button('パフォーマンスグラフを表示する', enable_events=True, key=f'{name}-Performance')],
        ]
        return sg.Frame(name, layout=layout, key=name)

    def load_config(self, config):
        self.config = config
        return

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

    def reset_slider(self, slot):
        length = slot.get_frame_length()
        self.window[f'{slot.name}-Slider'].update(range=(0,length), value=0)

    def update_slider(self, slot):
        if slot.isActive and slot.mode == 'video':
            i = slot.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.window[f'{slot.name}-Slider'].update(value=i)

    def play_pause(self, slot):
        if slot.isPlaying:
            self.window[f'{slot.name}-Play:Pause'].update('| |')
        else:
            self.window[f'{slot.name}-Play:Pause'].update('▶')

    def get_slot(self, name):
        for slot in self.slots:
            if slot.name == name:
                return slot
        else:
            return None

    def save_config(self, path):
        if self.config is not None:
            config = self.config
        else:
            config = {}
        for slot in self.slots:
            config.setdefault(slot.name, {})
            config[slot.name]['mode'] = slot.mode
            config[slot.name]['camera_id'] = slot.camera_id
            config[slot.name]['video_path'] = slot.video_path
        with open(path, 'w') as f:
            json.dump(config, f)
        return


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
            return True
        if event == '-Setting-':
            self.slot.open_setting()
        if event == '-OK-':
            self.close()
        if event is None:
            self.close()
        return False

from preprocess import Tuner

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
        self.Tuner = Tuner(num_samples=3)

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
            [sg.Button('背景差分パラメータを自動調節', enable_events=True, key='-AutoTune-'), sg.Text('', key='-Text-')],
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
        if event == '-AutoTune-':
            self.Tuner.activate(self.slot, self.window['-Text-'])
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

        if self.Tuner.active:
            self.Tuner.process()
            self.window['-a_min-'].update(value=self.slot.background_subtractor.a_min)
            self.window['-a_max-'].update(value=self.slot.background_subtractor.a_max)
            self.window['-c_min-'].update(value=self.slot.background_subtractor.c_min)
            self.window['-c_max-'].update(value=self.slot.background_subtractor.c_max)
            if self.Tuner.process_end:
                self.window['-a_min-'].update(value=self.slot.background_subtractor.a_min)
                self.window['-a_max-'].update(value=self.slot.background_subtractor.a_max)
                self.window['-c_min-'].update(value=self.slot.background_subtractor.c_min)
                self.window['-c_max-'].update(value=self.slot.background_subtractor.c_max)
                self.window['-Text-'].update(f'{self.Tuner.score:.1f}% match')
                self.Tuner.reset()
       
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


class CameraParameterWindow(BaseWindow):
    def __init__(self, settings_path):
        super().__init__()
        self.settings_path = settings_path
        self.slot = None
        self.camera_settings = None
        self.selected_setting = None

    def open(self, slot):
        self.slot = slot
        self.camera_settings = self.get_camera_settings()
        self.selected_setting = slot.camera_setting
        layout = [
            [sg.Text('load setting'), sg.Combo(list(self.camera_settings.keys()), default_value=self.get_default_value(slot), enable_events=True, key='-Select-')],
            [sg.Text('image size'), sg.Input(size=(5,1), key='w'), sg.Text('x'), sg.Input(size=(5,1), key='h')],
            [sg.Text('FOV'), sg.Input(size=(5,1), key='FOV')],
            [sg.Text('position'), sg.Text('x'), sg.Input(size=(5,1), key='pos_x'), sg.Text('y'), sg.Input(size=(5,1), key='pos_y'), sg.Text('z'), sg.Input(size=(5,1), key='pos_z')],
            [sg.Text('rotation'), sg.Text('x'), sg.Input(size=(5,1), key='rot_x'), sg.Text('y'), sg.Input(size=(5,1), key='rot_y'), sg.Text('z'), sg.Input(size=(5,1), key='rot_z')],
            [sg.Button('save setting', enable_events=True, key='-Save-')],
            [sg.Button('OK', enable_events=True, key='-OK-')]
        ]
        super().open('Camera Parameter Settings', layout=layout)

    def close(self):
        self.slot = None
        super().close()

    def get_camera_settings(self):
        with open(self.settings_path, 'r') as f:
            camera_settings = json.load(f)
        return camera_settings

    def get_default_value(self, slot):
        if slot.camera_setting is None:
            return ''
        else:
            return slot.camera_setting['name']

    def calc_proj_matrix(self):
        try:
            w = float(self.window['w'].get())
            h = float(self.window['h'].get())
            cx = w / 2
            cy = h / 2
            fov = float(self.window['FOV'].get())
            fx = (w/2)/np.tan(np.radians(fov/2))
            fy = (h/2)/np.tan(np.radians(fov/2))
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            px = float(self.window['pos_x'].get())
            py = float(self.window['pos_y'].get())
            pz = float(self.window['pos_z'].get())
            t = np.array([px,py,pz]).reshape([3,1])
            rx = np.radians(float(self.window['rot_x'].get()))
            ry = np.radians(float(self.window['rot_y'].get()))
            rz = np.radians(float(self.window['rot_z'].get()))
            R,_ = cv2.Rodrigues(np.array([rx,ry,rz]))
            Rc = R.T
            tc = -Rc @ t
            proj_matrix = K.dot(np.concatenate([Rc,tc],axis=1))
            print(proj_matrix)
            return proj_matrix
        except:
            print('cannot convert to projact matrix')
            return None

    def handle(self):
        event, values = self.window.read(timeout=0)
        if event == '-Select-':
            name = values[event]
            self.selected_setting = self.camera_settings[name]
        if event == '-Save-':
            self.calc_proj_matrix()
        if event == '-OK-':
            if self.selected_setting is not None:
                self.slot.camera_setting = self.selected_setting
                self.slot.proj_matrix = np.array(self.selected_setting['proj_matrix'])
            self.close()
        if event is None:
            self.close()