import os
import cv2
import numpy as np
from vlnce.vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlnce.vlfm.vlm.yolov7 import YOLOv7Client
from vlnce.vlfm.vlm.sam import MobileSAMClient
from vlnce.vlfm.vlm.blip2 import BLIP2Client
from vlnce.vlfm.vlm.blip2itm import BLIP2ITMClient
from vlnce.vlfm.vlm.coco_classes import COCO_CLASSES
from vlnce.vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap

from vlnce.utils.geometry_utils import get_fov

MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  # "table",
    "framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "cushion",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]

class ImageModels:
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
    ):
        self.object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self.coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self.mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self.blip2itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        
        self._use_vqa = use_vqa
        if use_vqa:
            self.vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))

        self._vqa_prompt = vqa_prompt

        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._target_object = "|".join(MP3D_ID_TO_NAME)
        # self._target_object = MP3D_ID_TO_NAME[1]
        self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."

        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=1)

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self.coco_object_detector.predict(img)
            if has_coco
            else self.object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self.object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections
    
    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self.mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.

            if self._use_vqa:
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self.vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue
                
            
            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections
    
    def visualize_detections(self, rgb, depth):
        """
        Visualizes the object detections on the RGB image.

        Args:
            rgb (np.ndarray): The RGB image to visualize the detections on.
            depth (np.ndarray): The depth image corresponding to the RGB image.
        """
        detections = self._get_object_detections(rgb)
        
        depth = depth / np.max(depth) * 255.0
        depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = rgb.copy()
            annotated_depth = depth.copy()

        viz_image = np.concatenate((annotated_rgb, annotated_depth), axis=1)

        # cv2.imshow("Object Detections", viz_image)
        # cv2.waitKey(1)
        return viz_image

    def _get_target_object_location(self, position: np.ndarray):
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None
