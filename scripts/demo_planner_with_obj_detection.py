import argparse
import os
import cv2
import time
import numpy as np

# omni-isaaclab
from omni.isaac.lab.app import AppLauncher

import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to collect data from the matterport dataset.")

parser.add_argument("--task", type=str, default="go2_matterport", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
# parser.add_argument("--draw_pointcloud", action="store_true", default=False, help="DRaw pointlcoud.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
# import ipdb; ipdb.set_trace()
simulation_app = app_launcher.app

import torch

import gymnasium as gym
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth


from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, RslRlVecEnvHistoryWrapper, VLNEnvWrapper

if __name__ == "__main__":
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)


    dataset_file_name = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    # scene_id = None

    init_pos = [15.068599700927734, 4.4848198890686035, 0.17162801325321198]
    init_rot = [0.25881901383399963, 0.0, 0.0, 0.9659258127212524]
    if "go2" in args_cli.task:
        env_cfg.scene.robot.init_state.pos = (init_pos[0], init_pos[1], init_pos[2]+0.4)
    else:
        raise ValueError(f"Task {args_cli.task} is not supported for this script.")
    env_cfg.scene.robot.init_state.rot = (init_rot[0], init_rot[1], init_rot[2], init_rot[3])
    env_cfg.scene_id = "zsNo4HB9uLZ"
    # env_cfg.scene_id = "YVUC4YcDtcY"

    print("scene_id: ", env_cfg.scene_id)
    
    udf_file = os.path.join(ASSETS_DIR, f"matterport_usd/{env_cfg.scene_id}/{env_cfg.scene_id}.usd")
    if os.path.exists(udf_file):
        env_cfg.scene.terrain.obj_filepath = udf_file
    else:
        raise ValueError(f"No USD file found in scene directory: {udf_file}")  

    print("scene_id: ", env_cfg.scene_id)
    print("robot_init_pos: ", env_cfg.scene.robot.init_state.pos)
    
    # initialize environment and low-level policy
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if args_cli.history_length > 0:
        env = RslRlVecEnvHistoryWrapper(env, history_length=args_cli.history_length)
    else:
        env = RslRlVecEnvWrapper(env)
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join(os.path.dirname(__file__),"../logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, agent_cfg.load_checkpoint)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)  # Adjust device as needed
    ppo_runner.load(resume_path)

    low_level_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    env = VLNEnvWrapper(env, low_level_policy, args_cli.task, high_level_obs_key="camera_obs", measure_names=None)

    # from vlnce.agents.teleop_agent import TeleopAgent
    # agent = TeleopAgent(env, simulation_app, vis_detections=False)

    # from vlnce.agents.vlfm_agent import VLFMAgent
    # agent = VLFMAgent(env, simulation_app)

    from vlnce.agents.mem3d_agent import Mem3DAgent
    agent = Mem3DAgent(env, simulation_app)

    agent.run_loop()
    # Close the simulator
    simulation_app.close()
    print("closed!!!")