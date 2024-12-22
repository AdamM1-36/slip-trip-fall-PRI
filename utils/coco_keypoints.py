from dataclasses import dataclass
from typing import List

import numpy as np

"""
0 - Nose
1 - Left Eye
2 - Right Eye
3 - Left Ear
4 - Right Ear
5 - Left Shoulder
6 - Right Shoulder
7 - Left Elbow
8 - Right Elbow
9 - Left Wrist
10 - Right Wrist
11 - Left Hip
12 - Right Hip
13 - Left Knee
14 - Right Knee
15 - Left Ankle
16 - Right Ankle
"""


@dataclass
class COCOKeypoints:
    person_index: int  # Tracker ID or person index
    nose_x: float
    nose_y: float
    nose_conf: float
    left_eye_x: float
    left_eye_y: float
    left_eye_conf: float
    right_eye_x: float
    right_eye_y: float
    right_eye_conf: float
    left_ear_x: float
    left_ear_y: float
    left_ear_conf: float
    right_ear_x: float
    right_ear_y: float
    right_ear_conf: float
    left_shoulder_x: float
    left_shoulder_y: float
    left_shoulder_conf: float
    right_shoulder_x: float
    right_shoulder_y: float
    right_shoulder_conf: float
    left_elbow_x: float
    left_elbow_y: float
    left_elbow_conf: float
    right_elbow_x: float
    right_elbow_y: float
    right_elbow_conf: float
    left_wrist_x: float
    left_wrist_y: float
    left_wrist_conf: float
    right_wrist_x: float
    right_wrist_y: float
    right_wrist_conf: float
    left_hip_x: float
    left_hip_y: float
    left_hip_conf: float
    right_hip_x: float
    right_hip_y: float
    right_hip_conf: float
    left_knee_x: float
    left_knee_y: float
    left_knee_conf: float
    right_knee_x: float
    right_knee_y: float
    right_knee_conf: float
    left_ankle_x: float
    left_ankle_y: float
    left_ankle_conf: float
    right_ankle_x: float
    right_ankle_y: float
    right_ankle_conf: float

    @classmethod
    def from_list(cls, person_index: int, keypoints_list: List[float]):
        if len(keypoints_list) != 51:
            raise ValueError(f"Expected 51 elements, got {len(keypoints_list)}")
        return cls(person_index, *keypoints_list)
    
    @staticmethod
    def keypoints_no_conf() -> list[str]:
        return [
            "nose_x",
            "nose_y",
            "left_eye_x",
            "left_eye_y",
            "right_eye_x",
            "right_eye_y",
            "left_ear_x",
            "left_ear_y",
            "right_ear_x",
            "right_ear_y",
            "left_shoulder_x",
            "left_shoulder_y",
            "right_shoulder_x",
            "right_shoulder_y",
            "left_elbow_x",
            "left_elbow_y",
            "right_elbow_x",
            "right_elbow_y",
            "left_wrist_x",
            "left_wrist_y",
            "right_wrist_x",
            "right_wrist_y",
            "left_hip_x",
            "left_hip_y",
            "right_hip_x",
            "right_hip_y",
            "left_knee_x",
            "left_knee_y",
            "right_knee_x",
            "right_knee_y",
            "left_ankle_x",
            "left_ankle_y",
            "right_ankle_x",
            "right_ankle_y",
        ]

def extract_keypoints(results):
    keypoints = []
    for result in results:
        if result.names[0] == "person":
            for obj, person_keypoints in zip(result.boxes, result.keypoints):
                try:
                    tracker_id = (
                        obj.id.item() if obj.id is not None else -1
                    )  # Assign -1 if no ID
                    points = person_keypoints.data.cpu().numpy().flatten().tolist()
                    keypoints.append(COCOKeypoints.from_list(tracker_id, points))
                except Exception as e:
                    print(f"Error extracting keypoints: {e}")
                    continue
    return keypoints
