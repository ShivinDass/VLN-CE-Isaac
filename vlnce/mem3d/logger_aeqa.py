import os
import json
import pickle
import logging
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image
from typing import Union

from vlnce.mem3d.tsdf_planner import TSDFPlanner, Frontier, SnapShot


class Logger:
    def __init__(
        self,
        output_dir,
        start_ratio,
        end_ratio,
        n_total_questions,
        voxel_size,  # used for calculating the moving distance
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size

        if os.path.exists(
            os.path.join(output_dir, f"success_list_{start_ratio}_{end_ratio}.pkl")
        ):
            with open(
                os.path.join(output_dir, f"success_list_{start_ratio}_{end_ratio}.pkl"),
                "rb",
            ) as f:
                self.success_list = pickle.load(f)
        else:
            self.success_list = []

        if os.path.exists(
            os.path.join(output_dir, f"path_length_list_{start_ratio}_{end_ratio}.pkl")
        ):
            with open(
                os.path.join(
                    output_dir, f"path_length_list_{start_ratio}_{end_ratio}.pkl"
                ),
                "rb",
            ) as f:
                self.path_length_list = pickle.load(f)
        else:
            self.path_length_list = {}

        if os.path.exists(
            os.path.join(output_dir, f"fail_list_{start_ratio}_{end_ratio}.pkl")
        ):
            with open(
                os.path.join(output_dir, f"fail_list_{start_ratio}_{end_ratio}.pkl"),
                "rb",
            ) as f:
                self.fail_list = pickle.load(f)
        else:
            self.fail_list = []

        if os.path.exists(
            os.path.join(output_dir, f"gpt_answer_{start_ratio}_{end_ratio}.json")
        ):
            with open(
                os.path.join(output_dir, f"gpt_answer_{start_ratio}_{end_ratio}.json"),
                "r",
            ) as f:
                self.gpt_answer_list = json.load(f)
        else:
            self.gpt_answer_list = []

        if os.path.exists(
            os.path.join(
                output_dir, f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir, f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json"
                ),
                "r",
            ) as f:
                self.n_filtered_snapshots_list = json.load(f)
        else:
            self.n_filtered_snapshots_list = {}

        if os.path.exists(
            os.path.join(
                output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}.json"
                ),
                "r",
            ) as f:
                self.n_total_snapshots_list = json.load(f)
        else:
            self.n_total_snapshots_list = {}

        if os.path.exists(
            os.path.join(output_dir, f"n_total_frames_{start_ratio}_{end_ratio}.json")
        ):
            with open(
                os.path.join(
                    output_dir, f"n_total_frames_{start_ratio}_{end_ratio}.json"
                ),
                "r",
            ) as f:
                self.n_total_frames_list = json.load(f)
        else:
            self.n_total_frames_list = {}

        self.n_total_questions = n_total_questions
        n_success = len(self.success_list)
        n_fail = len(self.fail_list)
        self.n_total = n_success + n_fail
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

        # some sanity check
        assert n_success == len(
            self.path_length_list
        ), f"{n_success} != {len(self.path_length_list)}"
        assert n_success == len(
            self.gpt_answer_list
        ), f"{n_success} != {len(self.gpt_answer_list)}"
        assert self.n_total == len(
            self.n_filtered_snapshots_list
        ), f"{self.n_total} != {len(self.n_filtered_snapshots_list)}"
        assert self.n_total == len(
            self.n_total_snapshots_list
        ), f"{self.n_total} != {len(self.n_total_snapshots_list)}"
        assert self.n_total == len(
            self.n_total_frames_list
        ), f"{self.n_total} != {len(self.n_total_frames_list)}"

        # logging for episode
        self.episode_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.explore_dist = 0

    def save_results(self):
        # sanity check
        assert len(self.success_list) == len(self.path_length_list)
        assert len(self.success_list) == len(self.gpt_answer_list)
        assert self.n_total == len(self.n_filtered_snapshots_list)
        assert self.n_total == len(self.n_total_snapshots_list)
        assert self.n_total == len(self.n_total_frames_list)

        with open(
            os.path.join(
                self.output_dir, f"success_list_{self.start_ratio}_{self.end_ratio}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_list, f)
        with open(
            os.path.join(
                self.output_dir,
                f"path_length_list_{self.start_ratio}_{self.end_ratio}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.path_length_list, f)
        with open(
            os.path.join(
                self.output_dir, f"fail_list_{self.start_ratio}_{self.end_ratio}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(self.fail_list, f)
        with open(
            os.path.join(
                self.output_dir, f"gpt_answer_{self.start_ratio}_{self.end_ratio}.json"
            ),
            "w",
        ) as f:
            json.dump(self.gpt_answer_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_filtered_snapshots_{self.start_ratio}_{self.end_ratio}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_filtered_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_snapshots_{self.start_ratio}_{self.end_ratio}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_frames_{self.start_ratio}_{self.end_ratio}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_frames_list, f, indent=4)

    def aggregate_results(self):
        # aggregate the results from different splits into a single file
        success_list = []
        path_length_list = {}
        all_success_list_paths = glob.glob(
            os.path.join(self.output_dir, "success_list_*.pkl")
        )
        all_path_length_list_paths = glob.glob(
            os.path.join(self.output_dir, "path_length_list_*.pkl")
        )
        for success_list_path in all_success_list_paths:
            with open(success_list_path, "rb") as f:
                success_list += pickle.load(f)
        for path_length_list_path in all_path_length_list_paths:
            with open(path_length_list_path, "rb") as f:
                path_length_list.update(pickle.load(f))

        with open(os.path.join(self.output_dir, "success_list.pkl"), "wb") as f:
            pickle.dump(success_list, f)
        with open(os.path.join(self.output_dir, "path_length_list.pkl"), "wb") as f:
            pickle.dump(path_length_list, f)

        gpt_answer_list = []
        all_gpt_answer_list_paths = glob.glob(
            os.path.join(self.output_dir, "gpt_answer_*.json")
        )
        for gpt_answer_list_path in all_gpt_answer_list_paths:
            with open(gpt_answer_list_path, "r") as f:
                gpt_answer_list += json.load(f)

        with open(os.path.join(self.output_dir, "gpt_answer.json"), "w") as f:
            json.dump(gpt_answer_list, f, indent=4)

        n_filtered_snapshots_list = {}
        all_n_filtered_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_filtered_snapshots_*.json")
        )
        for n_filtered_snapshots_list_path in all_n_filtered_snapshots_list_paths:
            with open(n_filtered_snapshots_list_path, "r") as f:
                n_filtered_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_filtered_snapshots.json"), "w") as f:
            json.dump(n_filtered_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of filtered snapshots: {np.mean(list(n_filtered_snapshots_list.values()))}"
        )

        n_total_snapshots_list = {}
        all_n_total_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_snapshots_*.json")
        )
        for n_total_snapshots_list_path in all_n_total_snapshots_list_paths:
            with open(n_total_snapshots_list_path, "r") as f:
                n_total_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_snapshots.json"), "w") as f:
            json.dump(n_total_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of total snapshots: {np.mean(list(n_total_snapshots_list.values()))}"
        )

        n_total_frames_list = {}
        all_n_total_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_frames_*.json")
        )
        for n_total_frames_list_path in all_n_total_frames_list_paths:
            with open(n_total_frames_list_path, "r") as f:
                n_total_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_frames.json"), "w") as f:
            json.dump(n_total_frames_list, f, indent=4)
        logging.info(
            f"Average number of total frames: {np.mean(list(n_total_frames_list.values()))}"
        )

    def log_episode_result(
        self,
        success: bool,
        question_id,
        explore_dist,
        gpt_answer,
        n_filtered_snapshots,
        n_total_snapshots,
        n_total_frames,
    ):
        if success:
            if question_id not in self.success_list:
                self.success_list.append(question_id)
            self.path_length_list[question_id] = explore_dist
            self.gpt_answer_list.append({"question_id": question_id, "answer": gpt_answer})
            logging.info(
                f"Question id {question_id} finish successfully, {explore_dist} length"
            )
        else:
            if question_id not in self.fail_list:
                self.fail_list.append(question_id)
            logging.info(f"Question id {question_id} failed, {explore_dist} length")

        logging.info(
            f"{self.n_total + 1}/{self.n_total_questions}: Success rate: {len(self.success_list)}/{self.n_total + 1}"
        )
        logging.info(
            f"Mean path length for success exploration: {np.mean(list(self.path_length_list.values()))}"
        )
        logging.info(
            f"Filtered snapshots/Total snapshots/Total frames: {n_filtered_snapshots}/{n_total_snapshots}/{n_total_frames}"
        )

        self.n_filtered_snapshots_list[question_id] = n_filtered_snapshots
        self.n_total_snapshots_list[question_id] = n_total_snapshots
        self.n_total_frames_list[question_id] = n_total_frames

        self.n_total += 1

        # clear up the episode log
        self.episode_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.explore_dist = 0

    def init_episode(
        self,
        question_id,
        init_pts_voxel,
    ):
        self.episode_dir = os.path.join(self.output_dir, question_id)
        eps_chosen_snapshot_dir = os.path.join(self.episode_dir, "chosen_snapshot")
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")
        eps_snapshot_dir = os.path.join(self.episode_dir, "snapshot")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_chosen_snapshot_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)
        os.makedirs(eps_snapshot_dir, exist_ok=True)

        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack([self.pts_voxels, init_pts_voxel])

        self.explore_dist = 0

        return (
            self.episode_dir,
            eps_chosen_snapshot_dir,
            eps_frontier_dir,
            eps_snapshot_dir,
        )

    def log_step(self, pts_voxel):
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        self.explore_dist += (
            np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
        )

    def save_topdown_visualization(self, cnt_step, fig):
        assert self.episode_dir is not None
        visualization_path = os.path.join(self.episode_dir, "visualization")
        os.makedirs(visualization_path, exist_ok=True)
        ax1 = fig.axes[0]
        ax1.plot(
            self.pts_voxels[:-1, 1], self.pts_voxels[:-1, 0], linewidth=1, color="white"
        )

        fig.tight_layout()
        plt.savefig(os.path.join(visualization_path, "{}_map.png".format(cnt_step)))
        plt.close()

    def save_frontier_visualization(
        self,
        cnt_step,
        tsdf_planner: TSDFPlanner,
        max_point_choice: Union[SnapShot, Frontier],
        global_caption,
    ):
        assert self.episode_dir is not None
        frontier_video_path = os.path.join(self.episode_dir, "frontier_video")
        episode_frontier_dir = os.path.join(self.episode_dir, "frontier")
        episode_snapshot_dir = os.path.join(self.episode_dir, "snapshot")
        os.makedirs(frontier_video_path, exist_ok=True)
        num_images = len(tsdf_planner.frontiers)
        if type(max_point_choice) == SnapShot:
            num_images += 1
        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if (i < num_images - 1) or (
                    i < num_images and type(max_point_choice) == Frontier
                ):
                    img_path = os.path.join(
                        episode_frontier_dir, tsdf_planner.frontiers[i].image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    if (
                        type(max_point_choice) == Frontier
                        and max_point_choice.image == tsdf_planner.frontiers[i].image
                    ):
                        axs[h_idx, w_idx].set_title("Chosen")
                elif i == num_images - 1 and type(max_point_choice) == SnapShot:
                    img_path = os.path.join(
                        episode_snapshot_dir, max_point_choice.image
                    )
                    img = matplotlib.image.imread(img_path)
                    axs[h_idx, w_idx].imshow(img)
                    axs[h_idx, w_idx].set_title("Snapshot Chosen")
        fig.suptitle(global_caption, fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(os.path.join(frontier_video_path, f"{cnt_step}.png"))
        plt.close()
