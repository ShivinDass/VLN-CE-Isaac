import cv2
import numpy as np
import math
import torch
from vlnce.utils.keyboard_reader import KeyboardReader
from vlnce.agents.base_agent import BaseAgent

from vlnce.utils.general_utils import extract_images, quat2eulers, visualize, is_large_orientation_change, add_angles
from vlnce.utils.geometry_utils import xyz_yaw_to_tf_matrix

from vlnce.utils.visualization import write_video
from vlnce.mem3d.semantic_map import SemanticMapping

QUESTION = "What is hanging from the oven handle?"

class Mem3DAgent(BaseAgent):
    def __init__(self, env, simulation_app, *args, **kwargs):
        super().__init__(env, simulation_app, *args, **kwargs)

        self.keyboardreader = KeyboardReader()

        self.forward_delta = 0.5  # Forward step size
        self.rotation_delta = np.pi/6  # Rotation step size in radians

        # # values coied over from vlfm if not easily available
        self.camera_intrinsics = self.env.unwrapped.scene["rgbd_camera"].data.intrinsic_matrices[0].detach().cpu().numpy()
        # self.fx = camera_intrinsics[0, 0]
        # self.fy = camera_intrinsics[1, 1]
        # camera_width = self.env.unwrapped.scene["rgbd_camera"].data.image_shape[1]
        # self.fov = 2 * math.atan2(camera_width / 2, self.fx)  # Field of view in radians
        
        self.reset()

    def reset(self):
        self.sim_steps = 0

        self.error_sum = 0.0
        self.prev_error = 0.0

        self.video = []

        self.semantic_map = SemanticMapping()
        self.tsdf_planner = None


    def get_action(self, obs, infos):
        action = ""
        key_pressed = self.keyboardreader.get_key()
        action = key_pressed

        return action
    
    def get_robot_pose_in_world(self):
        pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        rot = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
        
        return np.array(pos), np.array(rot)
    
    def get_camera_pose_in_world(self):
        camera_pos = self.env.unwrapped.scene["rgbd_camera"].data.pos_w[0].detach().cpu().numpy()
        camera_quat_w = self.env.unwrapped.scene["rgbd_camera"].data.quat_w_world[0].detach().cpu().numpy()
        camera_rot = quat2eulers(camera_quat_w[0], camera_quat_w[1], camera_quat_w[2], camera_quat_w[3])
        
        return np.array(camera_pos), np.array(camera_rot)
    
    def pid_control(self, current_position, target_position, dt=0.2, Kp=3.0, Ki=0.00, Kd=0.000):
        """PID controller to compute velocity correction."""
        # target_position = np.array([target_position[0]])
        error = target_position[0] - current_position[0]
        # print("error: ", error)
        self.error_sum += error * dt
        error_diff = (error - self.prev_error) / dt
        correction = Kp * error + Ki * self.error_sum + Kd * error_diff
        self.prev_error = error
        return correction
    
    def take_continuous_action(self, action, obs, infos):
        vel_command = np.zeros(3, dtype=np.float32)

        vel_step = 1.0
        if action == 'up':
            vel_command[0] = vel_step
        elif action == 'left':
            vel_command[2] = vel_step
        elif action == 'right':
            vel_command[2] = -vel_step
        elif action == 'space':
            print("Saving video...")
            write_video(self.video, "vlfm_isaac_video.mp4", fps=10)
            self.video = []

        obs, _, done, infos = self.env.step(vel_command)

        visualize(infos)

        return obs, 0, done, infos
    

    def get_observations(self, infos):
        obs = {}
        obs['rgb'], obs['depth'] = extract_images(infos, process_depth=False)

        cam_pos, cam_yaw = self.get_camera_pose_in_world()
        cam_yaw = cam_yaw[2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(cam_pos, cam_yaw)
        obs['cam_pose'] = tf_camera_to_episodic

        rob_pos, rob_yaw = self.get_robot_pose_in_world()
        obs['rob_pos'] = rob_pos

        return obs        

    def run_loop(self):
        robot_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        _, _, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
        cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        self.env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

        # Reset the environment and apply zero velocity command
        obs, infos = self.env.reset()
        
        print("Starting 3DMemAgent loop...")

        it = 0
        while self.simulation_app.is_running(): # 20hz

            action = self.get_action(obs, infos)

            obs, _, done, infos = self.take_continuous_action(action, obs, infos)
            
            if action != "" and it % 20 == 0:
                obs = self.get_observations(infos)

                self.semantic_map.update_scene_graph(
                        image_rgb=obs['rgb'],
                        depth=obs['depth'],
                        intrinsics=self.camera_intrinsics,
                        cam_pos=obs['cam_pose'],
                        pts=obs['rob_pos'],
                        pts_voxel=None,#self.tsdf_planner.habitat2voxel(pts),
                        img_path='random.png',
                        frame_idx=it,
                        target_obj_mask=None,
                    )
                self.visualize(obs, infos)

            it += 1

            if done:
                print("Episode done!!!")
                break

            self._observations_cache = {}
        
    def visualize(self, obs, infos):
        pass
        # rgb, depth = extract_images(infos, process_depth=False)

        # self.video.append(vis)
        # cv2.imshow("Annotated Maps", vis)
        # cv2.waitKey(1)


if __name__ == "__main__":
    agent = Mem3DAgent(None, None)

