import cv2
import math
import numpy as np

import omni.isaac.lab.utils.math as math_utils

def extract_images(infos, process_depth=True):
    rgb = infos['observations']['camera_obs'] # shape (1, 512, 512, 4)
    rgb = rgb[0,:,:,:3].clone().detach().cpu().numpy()
    # rgb = cv2.cvtColor(rgb[0,:,:,:3].clone().detach().cpu().numpy(), cv2.COLOR_RGB2BGR)

    depth = infos['observations']['depth_obs'] # shape (1, 512, 512, 1)
    depth = depth[0, 0,:,:, None].clone().detach().cpu().numpy()

    if process_depth:
        depth = depth / np.max(depth) * 255.0
        depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return rgb, depth

def visualize(infos):
    rgb, depth = extract_images(infos)
    viz_image = np.concatenate((rgb, depth), axis=1)
    
    cv2.imshow("visualization", viz_image)
    cv2.waitKey(1)

def is_large_orientation_change(env):
        
    robot_ori_full_quat = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().unsqueeze(0)
    robot_ori_full_rpy = math_utils.euler_xyz_from_quat(robot_ori_full_quat)

    for i_ori in range(2):
        if robot_ori_full_rpy[i_ori][0] > math.pi:
            robot_ori_full_rpy[i_ori][0] -= 2*math.pi
    
    # print("robot orientation: ", robot_ori_full_rpy)
    if abs(robot_ori_full_rpy[0].numpy()) > 0.6 or abs(robot_ori_full_rpy[1].numpy()) > 0.6:
        print("Large orientation: ", robot_ori_full_rpy[0], " ", robot_ori_full_rpy[1])
        return True
    return False

def quat2eulers(q0, q1, q2, q3):
    """
    Calculates the roll, pitch, and yaw angles from a quaternion.

    Args:
        q0: The scalar component of the quaternion.
        q1: The x-component of the quaternion.
        q2: The y-component of the quaternion.
        q3: The z-component of the quaternion.

    Returns:
        A tuple containing the roll, pitch, and yaw angles in radians.
    """

    roll = math.atan2(2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3), q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2)

    return roll, pitch, yaw

def add_angles(angle1, angle2):
    """
    Adds two angles in radians and normalizes the result to the range [-pi, pi].

    Args:
        angle1: The first angle in radians.
        angle2: The second angle in radians.

    Returns:
        The sum of the two angles, normalized to the range [-pi, pi].
    """
    result = angle1 + angle2
    if result > math.pi:
        result -= 2 * math.pi
    elif result < -math.pi:
        result += 2 * math.pi
    return result