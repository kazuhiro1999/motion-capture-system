from cProfile import run
import cv2
import PySimpleGUI as sg
from slot import Slot
from tools import Checkpoint, draw_keypoints2d
from window import CameraWindow, MainWindow, PerformanceWindow, SettingWindow
from core import Session, run_interface

crop_size = [224,224]

slot1 = Slot('Slot1')
slot2 = Slot('Slot2')
slots = [slot1, slot2]

window = MainWindow(slots)
camera_window = CameraWindow()
background_window = SettingWindow()
performance_window = PerformanceWindow()

session = Session('models/pose2d_mobile_gray_random_100.onnx', 'cpu')
checkpoint = Checkpoint()

window.open()

while True:
    
    checkpoint.reset_time()
    for slot in slots:
        ret, frame = slot.read()
        checkpoint.check(f'{slot.name}-read-image')
        if ret:
            image = frame
            if slot.subtract:
                image = slot.background_subtractor.apply(image, crop_region=slot.crop_region, crop_size=crop_size)
                checkpoint.check(f'{slot.name}-background-subtraction')
            if slot.pose_estimation:
                keypoints2d = run_interface(session, image) # shape:(N,1,17,3)
            if slot.draw_keypoints2d:
                image = draw_keypoints2d(image, keypoints2d[0,0])
            if slot.draw_image:
                cv2.imshow(slot.name, image)
                checkpoint.check(f'{slot.name}-draw-image')

    checkpoint.check('fps')

    event, values = window.read()
    slot, action = window.handle_events(event)

    if action == 'Close':
        break
    if action == 'Camera':
        camera_window.open(slot)
    if action == 'Video':
        video_path = sg.popup_get_file('動画を選択してください')
        slot.set_video(video_path)
    if action == 'Image':
        slot.draw_image = values[event]
    if action == 'Play:Pause':
        pass
    if action == 'Slider':
        pass
    if action == 'Subtraction':
        slot.subtract = values[event]
    if action == 'PoseEstimation':
        slot.pose_estimation = values[event]
    if action == 'Keypoints':
        slot.draw_keypoints2d = values[event]
    if action == 'Setting':
        background_window.open(slot)
    if action == 'Performance':
        performance_window.open(keys=checkpoint.keys)

    if camera_window.isActive:
        camera_window.handle()

    if background_window.isActive:
        background_window.handle()

    if performance_window.isActive:
        performance_window.update(checkpoint.get_results())
        event, values = performance_window.window.read(0)
        if event is None:
            performance_window.close()

for slot in slots:
    slot.close()
window.close()
camera_window.close()
background_window.close()
performance_window.close()