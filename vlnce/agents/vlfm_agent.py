import cv2
import numpy as np
import math
import torch
from vlnce.utils.keyboard_reader import KeyboardReader
from vlnce.vlfm.object_detector import ImageModels
from vlnce.vlfm.mapping.obstacle_map import ObstacleMap
from vlnce.vlfm.mapping.value_map import ValueMap

from vlnce.utils.general_utils import extract_images
from vlnce.agents.base_agent import BaseAgent
import omni.isaac.lab.utils.math as math_utils

from vlnce.utils.general_utils import quat2eulers, visualize, is_large_orientation_change, add_angles
from vlnce.utils.geometry_utils import xyz_yaw_to_tf_matrix

from vlnce.utils.visualization import write_video

PROMPT_SEPARATOR = "|"

class VLFMAgent(BaseAgent):
    def __init__(self, env, simulation_app, *args, **kwargs):
        super().__init__(env, simulation_app, *args, **kwargs)

        self.keyboardreader = KeyboardReader()

        self.image_models = ImageModels()

        self.forward_delta = 0.5  # Forward step size
        self.rotation_delta = np.pi/6  # Rotation step size in radians

        # values coied over from vlfm if not easily available
        camera_intrinsics = self.env.unwrapped.scene["rgbd_camera"].data.intrinsic_matrices[0].detach().cpu().numpy()
        self.fx = camera_intrinsics[0, 0]
        self.fy = camera_intrinsics[1, 1]
        camera_width = self.env.unwrapped.scene["rgbd_camera"].data.image_shape[1]
        self.fov = 2 * math.atan2(camera_width / 2, self.fx)  # Field of view in radians
        
        self.min_depth = 0.0  # Minimum depth value in meters
        self.max_depth = 100.0  # Maximum depth value in meters

        self.pixels_per_meter = 20  # Pixels per meter for the value map

        self.reset()

    def reset(self):
        self.sim_steps = 0

        self.error_sum = 0.0
        self.prev_error = 0.0

        self.obstacle_map = ObstacleMap(
            min_height=0.15,
            max_height=0.88,
            area_thresh=1.5,
            agent_radius=0.08,
            hole_area_thresh=100000,
            pixels_per_meter=self.pixels_per_meter,
        )
        self._observations_cache = {}

        self._text_prompt = "Seems like there is a target_object ahead."
        self._value_map: ValueMap = ValueMap(
            value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=False,
            obstacle_map=self.obstacle_map,
            pixels_per_meter=self.pixels_per_meter,
        )

        self.video = []

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
    
    def take_low_level_action(self, action, obs, infos):
        pos, rot = self.get_robot_pose_in_world()
        pos = pos[:2]
        rot = rot[2]

        target_pos = pos.copy()
        target_rot = rot
        if action == "up":
            target_pos[0] += self.forward_delta * math.cos(rot)
            target_pos[1] += self.forward_delta * math.sin(rot)
        elif action == "left":
            target_rot = add_angles(target_rot, self.rotation_delta)
        elif action == "right":
            target_rot = add_angles(target_rot, -self.rotation_delta)
        elif action == "space":
            rgb, depth = extract_images(infos, process_depth=False)
            self.image_models.visualize_detections(rgb, depth)
            return obs, 0, False, infos
        else:
            return obs, 0, False, infos

        # print("action: ", action)
        # print(f"Current Pose: {pos}, {rot}")
        # print(f"Target Pose: {target_pos}, {target_rot}")

        goal_reached_steps = 0
        max_steps = 50
        while True:
            
            vel = np.zeros(3)
            # calc position velocity
            relative_pos = torch.zeros(3, dtype=torch.float32, device=self.env.unwrapped.device)
            relative_pos[:2] = torch.tensor(target_pos - pos, dtype=torch.float32, device=self.env.unwrapped.device)
            
            expert_pos_body_frame = math_utils.quat_rotate_inverse(math_utils.yaw_quat(self.env.unwrapped.scene['robot'].data.root_quat_w), 
                                                         relative_pos.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            correction = self.pid_control(pos*0, expert_pos_body_frame)
            vel[0] = np.clip(correction, 0, 1)  # Forward velocity

            # calc yaw velocity
            yaw_diff = add_angles(target_rot, -rot)
            target_yaw_rate = np.clip(5.0 * yaw_diff, -1, 1)
            vel[2] = target_yaw_rate
            
            vel = torch.tensor(vel, dtype=torch.float32, device=self.env.unwrapped.device)
            obs, _, done, infos = self.env.step(vel)

            visualize(infos)
                
            if is_large_orientation_change(self.env):
                print("Large orientation change detected, stopping the loop.")
                break

            pos, rot = self.get_robot_pose_in_world()
            pos = pos[:2]
            rot = rot[2]
            if np.linalg.norm(pos - target_pos) < 0.1 and abs(add_angles(rot, -target_rot)) < 0.1:
                goal_reached_steps += 1
            
            if goal_reached_steps > 5:
                break
            if max_steps <=0:
                print("Max steps reached, stopping the loop.")
                break

            max_steps -= 1
            self.sim_steps += 1

        infos['max_steps_reached'] = max_steps <= 0
        return obs, 0, done, infos
    
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
        elif action == 'd':
            print('Interact with Depth')
            
            _, depth_viz = extract_images(infos, process_depth=True)
            _, depth = extract_images(infos, process_depth=False)
            
            cv2.namedWindow("Depth Visualization")

            def print_pixel_value(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    depth_value = depth[y, x, 0]
                    print(f"Depth at ({x}, {y}): {depth_value:.2f} m")

            while True:
                cv2.setMouseCallback("Depth Visualization", print_pixel_value)
                cv2.imshow("Depth Visualization", depth_viz)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        obs, _, done, infos = self.env.step(vel_command)

        visualize(infos)

        return obs, 0, done, infos
        
    def run_loop(self):
        robot_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        _, _, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
        cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        self.env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

        # Reset the environment and apply zero velocity command
        obs, infos = self.env.reset()
        self._pre_step(obs, infos)

        print("Starting VLFMAgent loop...")

        it = 0
        while self.simulation_app.is_running(): # 20hz

            action = self.get_action(obs, infos)

            # if not self._done_initializing:  # Initialize
            #     mode = "initialize"
            #     pointnav_action = self._initialize()
            # elif goal is None:  # Haven't found target object yet
            #     mode = "explore"
            #     pointnav_action = self._explore(observations)
            # else:
            #     mode = "navigate"
            #     pointnav_action = self._pointnav(goal[:2], stop=True)

            obs, _, done, infos = self.take_continuous_action(action, obs, infos)
            # obs, _, done, infos = self.take_low_level_action(action, obs, infos)
            
            if action != "" and it % 20 == 0:
                self._pre_step(obs, infos)
                self.visualize(obs, infos)
                # print(infos["observations"].keys())

            it += 1

            if done:
                print("Episode done!!!")
                break

            self._observations_cache = {}

    def visualize(self, obs, infos):
        rgb, depth = extract_images(infos, process_depth=False)
        viz_image = self.image_models.visualize_detections(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth)

        obstacle_map = cv2.cvtColor(self.obstacle_map.visualize(), cv2.COLOR_BGR2RGB)
        value_map = self._visualize_value()

        sizes = 512
        obstacle_map = cv2.resize(obstacle_map, dsize=(sizes, sizes))
        value_map = cv2.resize(value_map, dsize=(sizes, sizes))
        maps = np.concatenate((obstacle_map, value_map), axis=1)

        viz_image = cv2.resize(viz_image, dsize=(1024, int(1024*viz_image.shape[0]/viz_image.shape[1])))

        vis = np.concatenate((viz_image, maps), axis=0)

        self.video.append(vis)
        cv2.imshow("Annotated Maps", vis)
        cv2.waitKey(1)

    def _visualize_value(self):
        markers = []
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": 5,
                "thickness": 2,
                "color": (0, 0, 255),
            }
            markers.append((frontier[:2], marker_kwargs))

        val_map = self._value_map.visualize(markers, reduce_fn=lambda x: np.max(x, axis=-1))#, obstacle_map=self.obstacle_map)

        return val_map
    
    def _pre_step(self, obs, infos):
        if len(self._observations_cache) > 0:
            return
        
        rgb, depth = extract_images(infos, process_depth=False)
        depth = depth[..., 0]

        # normalize depth
        depth = np.clip(depth, self.min_depth, self.max_depth)
        depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        
        rob_pos, rob_yaw = self.get_robot_pose_in_world()
        rob_yaw = rob_yaw[2]

        cam_pos, cam_yaw = self.get_camera_pose_in_world()
        cam_yaw = cam_yaw[2]

        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(cam_pos, cam_yaw)

        self.obstacle_map.update_map(
            depth=depth,
            tf_camera_to_episodic=tf_camera_to_episodic,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            fx=self.fx,
            fy=self.fy,
            topdown_fov=self.fov,
        )

        frontiers = self.obstacle_map.frontiers
        self.obstacle_map.update_agent_traj(rob_pos[:2], rob_yaw)

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "robot_xy": rob_pos[:2],
            "robot_heading": rob_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self.min_depth,
                    self.max_depth,
                    self.fx,
                    self.fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self.min_depth,
                    self.max_depth,
                    self.fov,
                )
            ],
        }
        
        self._update_value_map()

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self.image_models._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]

        goal = self.image_models._get_target_object_location(rob_pos[:2])
    
    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self.image_models.blip2itm.cosine(
                    rgb,
                    p.replace("target_object", self.image_models._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )
