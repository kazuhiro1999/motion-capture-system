from tkinter import E
import cv2
import PySimpleGUI as sg
from slot import Slot
from tools import Checkpoint, draw_crop_area, draw_keypoints2d
from triangulate import NoiseRemover, Triangulator
from udp_server import UDPServer, keypoints3d_to_data
from window import CameraParameterWindow, CameraWindow, MainWindow, PerformanceWindow, SettingWindow
from core import Session, run_interface
from detector import transform_keypoints, determine_crop_region

crop_size = [224,224]

slot1 = Slot('Slot1')
slot2 = Slot('Slot2')
slot3 = Slot('Slot3')
slot4 = Slot('Slot4')
slots = [slot1, slot2, slot3, slot4]

window = MainWindow(slots)
camera_window = CameraWindow()
background_window = SettingWindow()
parameter_window = CameraParameterWindow(settings_path='camera_settings.json')
performance_window = PerformanceWindow()

session = Session('models/pose2d_mobile_gray_random_100.onnx', 'gpu')
triangulator = Triangulator(slots)
remover = NoiseRemover(history=5, weights=None)
udp_server = UDPServer(host='127.0.0.1', port=2435)

checkpoint = Checkpoint()

window.open()

while True:
    
    checkpoint.reset_time()
    for slot in slots:
        if slot.isActive:
            ret, frame = slot.read()
            checkpoint.check(f'{slot.name}-read-image')
            if not ret:
                continue
            window.update_slider(slot)

            image = frame
            H,W = image.shape[:2]
            # 背景差分
            if slot.subtract:
                #image = slot.background_subtractor.detect_person(image, crop_region=slot.crop_region, crop_size=crop_size)
                image = slot.background_subtractor.apply(image, crop_region=slot.crop_region, crop_size=crop_size)
                checkpoint.check(f'{slot.name}-background-subtraction')
            # 姿勢推定
            if slot.pose_estimation:
                keypoints2d = run_interface(session, image) # shape:(N,1,17,3)
                slot.keypoints2d = transform_keypoints(keypoints2d, H, W, slot.crop_region, crop_size)
                checkpoint.check(f'{slot.name}-pose-estimation')
                slot.crop_region = determine_crop_region(slot.keypoints2d, H, W)
            # 描画
            if slot.draw_crop_area:
                if slot.draw_keypoints2d:
                    image = frame.copy()
                    image = draw_keypoints2d(image, slot.keypoints2d, s=5)
                    image = draw_crop_area(image, slot.crop_region)
                if slot.draw_image:
                    if slot.image_size is not None:
                        image = cv2.resize(image, dsize=slot.image_size)
                    cv2.imshow(slot.name, image)
            else:  
                if slot.draw_keypoints2d:
                    image = draw_keypoints2d(image, keypoints2d[0,0])
                if slot.draw_image:
                    if slot.image_size is not None and not slot.subtract:
                        image = cv2.resize(image, dsize=slot.image_size)
                    cv2.imshow(slot.name, image)

    # 3次元復元
    keypoints3d = None
    if triangulator.isActive:
        keypoints3d = triangulator.triangulate()
    # 時系列平滑化
    if remover.isActive:
        keypoints3d = remover.apply(keypoints3d)
    # UDP送信
    if udp_server.isActive:
        data = keypoints3d_to_data(keypoints3d)
        udp_server.send(data)

    checkpoint.check('execution')

    # メインウィンドウのGUI処理
    event, values = window.read()
    slot, action = window.handle_events(event)

    if action == 'Close':
        break
    if action == 'Camera':
        camera_window.open(slot)
    if action == 'Video':
        video_path = sg.popup_get_file('動画を選択してください')
        slot.set_video(video_path)
        window.reset_slider(slot)
        window.play_pause(slot)
    if action == 'Image':
        slot.draw_image = values[event]
    if action == 'Size':
        slot.set_size(values[event])
    if action == 'Play:Pause':
        if slot is not None:
            slot.isPlaying = not slot.isPlaying
            window.play_pause(slot)
        else:
            for slot in slots:
                slot.isPlaying = not slot.isPlaying
                window.play_pause(slot)
    if action == 'Slider':
        slot.set_frame(values[event])
    if action == 'Subtraction':
        slot.subtract = values[event]
    if action == 'CropArea':
        slot.draw_crop_area = values[event]
    if action == 'PoseEstimation':
        slot.pose_estimation = values[event]
    if action == 'Keypoints':
        slot.draw_keypoints2d = values[event]
    if action == 'Setting':
        background_window.open(slot)
    if action == 'ProjMatrix':
        parameter_window.open(slot)
    if action == 'Performance':
        performance_window.open(keys=checkpoint.keys)
    if action == 'Triangulation':
        triangulator.isActive = values[event]
    if action == 'Smoothing':
        remover.isActive = values[event]
    if action == 'UDP':
        udp_server.isActive = values[event]

    if camera_window.isActive:
        isSelected = camera_window.handle()
        if isSelected:
            window.reset_slider(camera_window.slot)
            window.play_pause(camera_window.slot)

    if background_window.isActive:
        background_window.handle()

    if parameter_window.isActive:
        parameter_window.handle()

    if performance_window.isActive:
        performance_window.update(checkpoint.get_results())
        event, values = performance_window.window.read(0)
        if event is None:
            performance_window.close()

window.save_config('config.json')
for slot in slots:
    slot.close()
window.close()
camera_window.close()
background_window.close()
performance_window.close()