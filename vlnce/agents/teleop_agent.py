import cv2
import numpy as np
from vlnce.utils.keyboard_reader import KeyboardReader
from vlnce.vlfm.object_detector import ImageModels

from vlnce.utils.general_utils import extract_images
from vlnce.agents.base_agent import BaseAgent
from vlnce.utils.visualization import write_video

import math
import torch
from vlnce.utils.general_utils import quat2eulers, visualize, is_large_orientation_change

class TeleopAgent(BaseAgent):
    def __init__(self, env, simulation_app, vis_detections=False, *args, **kwargs):
        super().__init__(env, simulation_app, *args, **kwargs)

        self.keyboardreader = KeyboardReader()
        self.vis_detections = vis_detections

        if vis_detections:
            self.object_detector = ImageModels()

        self.step_count = 0
        
        self.video = []

    def run_loop(self):
        robot_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        _, _, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
        cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        self.env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

        # Reset the environment and apply zero velocity command
        obs, infos = self.env.reset()

        it = 0
        while self.simulation_app.is_running(): # 20hz
            visualize(infos)
            
            if is_large_orientation_change(self.env):
                print("Large orientation change detected, stopping the loop.")
                break

            action = self.get_action(obs, infos)
            action = torch.tensor(action, dtype=torch.float32, device=self.env.unwrapped.device)
            obs, _, done, infos = self.env.step(action)

            if done:
                print("Episode done!!!")
                break
            
            it += 1

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
    
