
from vlnce.utils.keyboard_reader import KeyboardReader
from vlnce.vlfm.object_detector import ObjectDetector

class Planner:
    def __init__(self, env, env_cfg, args_cli, simulation_app):
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.simulation_app = simulation_app

        self.robot_start_pos = None

        if use_image_model:
            self.object_detector = ObjectDetector()
        self.keyboardreader = KeyboardReader()

    def pid_control(self, current_position, target_position, error_sum, prev_error, dt=0.2, Kp=0.5, Ki=0.00, Kd=0.000):
        """PID controller to compute velocity correction."""
        error = target_position[0] - current_position[0]
        error_sum += error * dt
        error_diff = (error - prev_error) / dt
        correction = Kp * error + Ki * error_sum + Kd * error_diff
        prev_error = error
        return correction, error_sum, prev_error
    
    def compute_target_yaw(self, current_position, next_position):
        """Compute the target yaw angle based on the current and next position."""
        delta_position = next_position - current_position
        target_yaw = np.arctan2(delta_position[1], delta_position[0])
        return (target_yaw)%(2*np.pi)

    def start_loop(self):
        """Start the simulation loop."""

        # Set the camera view
        robot_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        robot_quat_w = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        _, _, yaw = quat2eulers(robot_quat_w[0], robot_quat_w[1], robot_quat_w[2], robot_quat_w[3])
        cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw), robot_pos_w[1] - 0.8 * math.cos(-yaw), robot_pos_w[2] + 0.8)
        cam_target = (robot_pos_w[0], robot_pos_w[1], robot_pos_w[2])
        self.env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

        # Reset the environment and apply zero velocity command
        obs, infos = self.env.reset()

        # Simulate physics
        it = 1
        self.vel_command = torch.tensor([0.0, 0.0, 0.0], device=self.env.unwrapped.device)

        while self.simulation_app.is_running(): # 20hz
            rgb_image = infos['observations']['camera_obs'] # shape (1, 512, 512, 4)
            rgb_image = cv2.cvtColor(rgb_image[0,:,:,:3].clone().detach().cpu().numpy(), cv2.COLOR_RGB2BGR)

            depth_image = infos['observations']['depth_obs'] # shape (1, 512, 512, 1)
            depth_image = depth_image[0, 0,:,:, None].clone().detach().cpu().numpy()

            visualize(rgb_image.copy(), depth_image.copy())

            robot_pos_w = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            
            robot_ori_full_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().unsqueeze(0)
            robot_ori_full_rpy = math_utils.euler_xyz_from_quat(robot_ori_full_quat)

            for i_ori in range(2):
                if robot_ori_full_rpy[i_ori][0] > math.pi:
                    robot_ori_full_rpy[i_ori][0] -= 2*math.pi
            
            # print("robot orientation: ", robot_ori_full_rpy)
            if abs(robot_ori_full_rpy[0].numpy()) > 0.6 or abs(robot_ori_full_rpy[1].numpy()) > 0.6:
                print("Large orientation: ", robot_ori_full_rpy[0], " ", robot_ori_full_rpy[1])
                return
            
            key_pressed = self.keyboardreader.get_key()
            vel_step = 2.0
            if key_pressed == 'up':
                self.vel_command[0] = torch.tensor(vel_step, device=self.env.unwrapped.device)
            elif key_pressed == 'left':
                self.vel_command[2] = torch.tensor(vel_step, device=self.env.unwrapped.device)
            elif key_pressed == 'right':
                self.vel_command[2] = torch.tensor(-vel_step, device=self.env.unwrapped.device)
            elif key_pressed == 'space':
                if use_image_model:
                    self.object_detector.visualize_detections(rgb_image, depth_image)
            elif key_pressed == None:
                self.vel_command[0] = torch.tensor(0.0, device=self.env.unwrapped.device)
                self.vel_command[2] = torch.tensor(0.0, device=self.env.unwrapped.device)

            print(self.vel_command)
            print()
            
            obs, _, done, infos = self.env.step(self.vel_command)

            if done:
                print("Episode done!!!")
                break
            
            it += 1

        # Print measurements
        print("\n============================== Episode Measurements ==============================")
        for key, value in infos["measurements"].items():
            print(f"{key}: {value}")