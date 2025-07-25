import math
import torch
from vlnce.utils.general_utils import quat2eulers, visualize, is_large_orientation_change

class BaseAgent:

    def __init__(self, env, simulation_app):
        self.env = env
        self.simulation_app = simulation_app
    
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
        raise NotImplementedError("This method should be overridden by subclasses.")
    