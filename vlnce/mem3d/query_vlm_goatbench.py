import logging
from typing import Tuple, Optional, Union
import random
import numpy as np

from vlnce.mem3d.eval_utils_gpt_goatbench import explore_step
from vlnce.mem3d.utils import resize_image
from vlnce.mem3d.tsdf_planner import TSDFPlanner, SnapShot, Frontier
from vlnce.mem3d.scene_goatbench import Scene


def query_vlm_for_response(
    subtask_metadata: dict,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    verbose: bool = False,
) -> Optional[Tuple[Union[SnapShot, Frontier], int]]:
    # prepare input for vlm
    step_dict = {}

    # prepare snapshots
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["snapshot_objects"] = {}
    step_dict["snapshot_imgs"] = {}
    step_dict["use_full_obj_list"] = cfg.use_full_obj_list
    for rgb_id, snapshot in scene.snapshots.items():
        resized_rgb = resize_image(
            scene.all_observations[rgb_id], cfg.prompt_h, cfg.prompt_w
        )

        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
        step_dict["snapshot_imgs"][rgb_id] = {
            "full_img": resized_rgb,
            "object_crop": [],
        }

        # crop the snapshot to contain only the objects in the snapshot
        if cfg.use_full_obj_list:
            selected_bbox_idx = [
                idx
                for idx in range(len(snapshot.visual_prompt))
                if snapshot.visual_prompt[idx].data["obj_id"][0]
                in snapshot.full_obj_list.keys()
            ]
        else:
            selected_bbox_idx = [
                idx
                for idx in range(len(snapshot.visual_prompt))
                if snapshot.visual_prompt[idx].data["obj_id"][0] in snapshot.cluster
            ]
        selected_bbox = snapshot.visual_prompt[selected_bbox_idx].xyxy.copy()
        selected_obj_ids = [
            snapshot.visual_prompt[idx].data["obj_id"][0] for idx in selected_bbox_idx
        ]

        # scale the bbox
        H, W = scene.all_observations[rgb_id].shape[:2]
        scale_h, scale_w = cfg.prompt_h / H, cfg.prompt_w / W
        selected_bbox[:, 0] *= scale_w
        selected_bbox[:, 2] *= scale_w
        selected_bbox[:, 1] *= scale_h
        selected_bbox[:, 3] *= scale_h
        selected_bbox = selected_bbox.astype(int)

        # get the image crop for each object
        for obj_id, bbox in zip(selected_obj_ids, selected_bbox):
            step_dict["snapshot_imgs"][rgb_id]["object_crop"].append(
                {
                    "obj_class": object_id_to_name[obj_id],
                    "obj_id": obj_id,
                    "crop": resized_rgb[bbox[1] : bbox[3], bbox[0] : bbox[2]],
                }
            )

    # prepare frontier
    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in tsdf_planner.frontiers
    ]

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]

    # query vlm
    (
        outputs,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        reason,
        n_filtered_snapshots,
    ) = explore_step(step_dict, cfg, verbose=verbose)
    if outputs is None:
        logging.error(f"explore_step failed and returned None")
        return None
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    # parse returned results
    try:
        target_type, target_index = outputs.split(",")[0].strip().split(" ")
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, failed!")
        return None

    if target_type not in ["snapshot", "frontier"]:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    if target_type == "snapshot":
        if int(target_index) < 0 or int(target_index) >= len(snapshot_id_mapping):
            logging.info(
                f"Target index can not match real objects: {target_index}, failed!"
            )
            return None
        target_index = snapshot_id_mapping[int(target_index)]
        logging.info(f"The index of target snapshot {target_index}")

        # get the target snapshot
        if target_index < 0 or target_index >= len(scene.snapshots):
            logging.info(
                f"Predicted snapshot target index out of range: {target_index}, failed!"
            )
            return None

        pred_target_snapshot = list(scene.snapshots.values())[target_index]
        pred_target_snapshot_id = list(scene.snapshots.keys())[target_index]
        logging.info(f"Next choice: Snapshot of {pred_target_snapshot.image}")

        # get the object choice
        object_choice, object_choice_id = outputs.split(",")[1].strip().split(" ")
        if object_choice != "object":
            logging.info(f"Invalid object choice: {object_choice}, failed!")
            return None
        object_choice_id = int(object_choice_id)
        if object_choice_id < 0 or object_choice_id >= len(
            snapshot_crop_mapping[pred_target_snapshot_id]
        ):
            logging.info(f"Object choice out of range: {object_choice_id}, failed!")
            return None
        object_choice_id = snapshot_crop_mapping[pred_target_snapshot_id][
            object_choice_id
        ]
        pred_target_obj_id = step_dict["snapshot_imgs"][pred_target_snapshot_id][
            "object_crop"
        ][object_choice_id]["obj_id"]
        logging.info(
            f"Next choice Object: {pred_target_obj_id}, {scene.objects[pred_target_obj_id]['class_name']}"
        )

        # instantiate a snapshot object that contains only the predicted object as the navigation target
        max_point_choice = SnapShot(
            image=pred_target_snapshot.image,
            color=(random.random(), random.random(), random.random()),
            obs_point=np.empty(3),
            full_obj_list={
                pred_target_obj_id: scene.objects[pred_target_obj_id]["conf"]
            },
            cluster=[pred_target_obj_id],
        )

        return max_point_choice, n_filtered_snapshots
    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(tsdf_planner.frontiers):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, failed!"
            )
            return None
        target_point = tsdf_planner.frontiers[target_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        pred_target_frontier = tsdf_planner.frontiers[target_index]

        return pred_target_frontier, n_filtered_snapshots
