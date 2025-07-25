# VLN-CE-Isaac Benchmark

## Overview
The VLN-CE-Isaac Benchmark is a framework for evaluating Visual Language Navigation in Isaac Lab. This repository contains the code and instructions to set up the environment, download the required data, and run the benchmark.

<p align="center">
  <img width="65%" src="./src/teaser.gif" alt="VLN-CE with Go2">
</p>

## TODO List
- [ ] Release the VLA example and evaluation code


## Installation

### Prerequisites
- Ubuntu 20.04 or higher
- NVIDIA GPU with CUDA support (check [here](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html) for more detailed requirements)

### Steps

1. Create a virtual environment with python 3.10:
    ```sh
    conda create -n vlnce-isaac python=3.10
    conda activate vlnce-isaac
    ```

2. Make sure that Isaac Sim is installed on your machine. Otherwise follow [this guideline](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install it. If installing via the Omniverse Launcher, please ensure that Isaac Sim 4.1.0 is selected and installed. On Ubuntu 22.04 or higher, you could install it via pip:
    ```sh
    pip install isaacsim-rl==4.1.0 isaacsim-replicator==4.1.0 isaacsim-extscache-physics==4.1.0 isaacsim-extscache-kit-sdk==4.1.0 isaacsim-extscache-kit==4.1.0 isaacsim-app==4.1.0 --extra-index-url https://pypi.nvidia.com
    ```

3. Clone Isaac Lab and link the extensions.

    **Note**: This codebase was tested with Isaac Lab 1.1.0 and may not be compatible with newer versions. Please make sure to use the modified version of Isaac Lab provided below, which includes important bug fixes and updates. As Isaac Lab is under active development, we will consider supporting newer versions in the future.

    ```shell
    git clone https://github.com/yang-zj1026/IsaacLab.git
    cd IsaacLab
    cd source/extensions
    ln -s {THIS_REPO_DIR}/isaaclab_exts/omni.isaac.vlnce .
    ln -s {THIS_REPO_DIR}/isaaclab_exts/omni.isaac.matterport .
    cd ../..
    ```

4. Run the Isaac Lab installer script and additionally install rsl rl in this repo.
    ```shell
    ./isaaclab.sh -i none
    ./isaaclab.sh -p -m pip install -e {THIS_REPO_DIR}/scripts/rsl_rl
    cd ..
    ```

## Data

Download the data from [huggingface](https://huggingface.co/datasets/Zhaojing/VLN-CE-Isaac) and put them under `isaaclab_exts/omni.isaac.vlnce/assets` directory.
The expected file structure should be like:
```graphql
isaaclab_exts/omni.isaac.vlnce
├─ assets
|   ├─ vln_ce_isaac_v1.json.gz
|   ├─ matterport_usd
```

## Code Usage

Run the demo with a PD path planner
```shell
python scripts/demo_planner.py --task=go2_matterport_vision --history_length=9 --load_run=2024-09-25_23-22-02

python scripts/demo_planner.py --task=h1_matterport_vision --load_run=2024-11-03_15-08-09_height_scan_obst
```
To train your own low-level policies, please refer to the [legged-loco](https://github.com/yang-zj1026/legged-loco) repo.

## Citation
If you use VLN-CE-Isaac in your work please consider citing our paper:
```
@article{cheng2024navila,
  title={NaVILA: Legged Robot Vision-Language-Action Model for Navigation},
  author={Cheng, An-Chieh and Ji, Yandong and Yang, Zhaojing and Zou, Xueyan and Kautz, Jan and B{\i}y{\i}k, Erdem and Yin, Hongxu and Liu, Sifei and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2412.04453},
  year={2024}
}
```

## Acknowledgements

This project makes use of the following open-source codebases:
- Isaac Lab: [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
- ViPlanner: [https://github.com/leggedrobotics/viplanner](https://github.com/leggedrobotics/viplanner)
- VLN-CE: [https://github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE)


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


```
# terminal 1
cd ~/Desktop/vlfm_habitat/vlfm
conda activate vlfm
./scripts/launch_gdino_servers.sh

```

```
# terminal 2
cd ~/Desktop/vlnce_isaac/VLN-CE-Isaac
conda activate vlnce-isaac
python scripts/demo_planner_with_obj_detection.py --task=go2_matterport_vision --history_length=9 --load_run=2024-09-25_23-22-02
```