import cv2
import numpy as np
from vlnce.utils.keyboard_reader import KeyboardReader
from vlnce.vlfm.object_detector import ImageModels

from vlnce.utils.general_utils import extract_images
from vlnce.agents.base_agent import BaseAgent
from vlnce.utils.visualization import write_video

class TeleopAgent(BaseAgent):
    def __init__(self, env, simulation_app, vis_detections=False, *args, **kwargs):
        super().__init__(env, simulation_app, *args, **kwargs)

        self.keyboardreader = KeyboardReader()
        self.vis_detections = vis_detections

        if vis_detections:
            self.object_detector = ImageModels()

        self.step_count = 0
        
        self.video = []

    def get_action(self, obs, infos):
        vel_command = np.zeros(3, dtype=np.float32)

        key_pressed = self.keyboardreader.get_key()
        vel_step = 1.0
        if key_pressed == 'up':
            vel_command[0] = vel_step
        elif key_pressed == 'left':
            vel_command[2] = vel_step
        elif key_pressed == 'right':
            vel_command[2] = -vel_step
        elif key_pressed == 'space':
            write_video(self.video, "teleop_video.mp4", fps=10)
            # if self.vis_detections:
            #     rgb, depth = extract_images(infos, process_depth=False)
            #     vis_img = self.object_detector.visualize_detections(rgb, depth)
            #     cv2.imshow("Object Detections", vis_img)
            #     cv2.waitKey(1)
        elif key_pressed == None:
            vel_command[0] = 0
            vel_command[2] = 0

        if self.step_count % 5 == 0:
            rgb, depth = extract_images(infos, process_depth=False)
            # vis_img = self.object_detector.visualize_detections(rgb, depth)
            
            vis_camera_obs = infos["observations"]["viz_camera_obs"][0,:,:,:3].clone().detach().cpu().numpy()

            vis_img = vis_camera_obs # np.concatenate((vis_camera_obs, vis_img), axis=1)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            self.video.append(vis_img)
            cv2.imshow("Object Detections", vis_img)
            cv2.waitKey(1)
        self.step_count += 1

        return vel_command
    
