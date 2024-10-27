import cv2
import math
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
from test_draw import draw_keypoints_and_skeleton
from utils.coco_keypoints import COCOKeypoints

WEIGHT = "yolo11n-pose.pt"
pose_model = YOLO(WEIGHT)
fall_model = tf.keras.models.load_model("ml/fall_detection_model.keras")

video_path = "fall_dataset/videos/video_1.mp4"
cap = cv2.VideoCapture(video_path)
print(f"video fps is {cap.get(cv2.CAP_PROP_FPS)}")


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


def check_fall_detection(
    x_min,
    y_min,
    x_max,
    y_max,
    left_shoulder_y,
    left_foot_y,
    left_body_y,
    len_factor,
    right_shoulder_y,
    right_foot_y,
    right_body_y,
):

    # Checking person for "horizontal" orientation
    dy = int(y_max) - int(y_min)
    dx = int(x_max) - int(x_min)
    difference = dy - dx

    if any(
        kp == 0
        for kp in [
            left_shoulder_y,
            left_foot_y,
            len_factor,
            left_body_y,
            right_shoulder_y,
            right_foot_y,
            right_body_y,
        ]
    ):
        print("Some keypoints are missing")
        return False

    # Calculate conditions, source: https://github.com/Y-B-Class-Projects/Human-Fall-Detection
    left_condition = (
        left_shoulder_y > left_foot_y - len_factor
        and left_body_y > left_foot_y - (len_factor / 2)
        and left_shoulder_y > left_body_y - (len_factor / 2)
    )

    right_condition = (
        right_shoulder_y > right_foot_y - len_factor
        and right_body_y > right_foot_y - (len_factor / 2)
        and right_shoulder_y > right_body_y - (len_factor / 2)
    )

    return left_condition or right_condition or difference < 0


def fall_detection_ml(box, keypoint):
    box = box.detach().cpu().numpy()
    keypoint = keypoint.detach().cpu().numpy()

    x_min, y_min, x_max, y_max = map(int, box[:4])

    keypoints = COCOKeypoints.from_list(keypoint.flatten().tolist())

    len_factor = math.sqrt(
        (
            (keypoints.left_shoulder_y - keypoints.left_hip_y) ** 2
            + (keypoints.left_shoulder_x - keypoints.left_hip_x) ** 2
        )
    )

    keypoint_list = [
        keypoints.left_shoulder_y,
        keypoints.left_shoulder_x,
        keypoints.right_shoulder_y,
        keypoints.right_shoulder_x,
        keypoints.left_hip_y,
        keypoints.left_hip_x,
        keypoints.right_hip_y,
        keypoints.right_hip_x,
        len_factor,
        keypoints.left_knee_y,
        keypoints.left_knee_x,
        keypoints.right_knee_y,
        keypoints.right_knee_x,
        keypoints.left_ankle_y,
        keypoints.right_ankle_y,
    ]
    keypoint_array = np.array(keypoint_list)

    fall_prediction = fall_model.predict(keypoint_array.reshape(1, 15))

    if fall_prediction == 1:
        return True, (x_min, y_min, x_max, y_max)
    return False, None


def fall_detection(box, keypoint):
    box.detach().cpu().numpy()
    keypoint.detach().cpu().numpy()

    x_min, y_min, x_max, y_max = map(int, box[:4])

    keypoints = COCOKeypoints.from_list(keypoint.flatten().tolist())

    len_factor = math.sqrt(
        (
            (keypoints.left_shoulder_y - keypoints.left_hip_y) ** 2
            + (keypoints.left_shoulder_x - keypoints.left_hip_x) ** 2
        )
    )

    fall_detected = check_fall_detection(
        x_min,
        y_min,
        x_max,
        y_max,
        keypoints.left_shoulder_y,
        keypoints.left_ankle_y,
        keypoints.left_hip_y,
        len_factor,
        keypoints.right_shoulder_y,
        keypoints.right_ankle_y,
        keypoints.right_hip_y,
    )
    if fall_detected:
        return True, (x_min, y_min, x_max, y_max)
    return False, None


def run_ml():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        results = pose_model.predict(frame, stream=False)
        for result in results:
            if result.boxes.data.size(0) == 0:
                continue
            for i in range(result.boxes.data.size(0)):
                boxes = result.boxes.data[i]
                keypoints = result.keypoints.data[i]
                draw_keypoints_and_skeleton(frame, keypoints)
                fall_detected, bbox = fall_detection_ml(boxes, keypoints)
                if fall_detected:
                    print("Person fell down")
                    x_min, y_min, x_max, y_max = bbox[0:4]
                    cv2.rectangle(
                        frame,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        color=(0, 0, 255),
                        thickness=5,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "Person Fell down",
                        (11, 100),
                        0,
                        1,
                        [0, 0, 255],
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )

        cv2.imshow("Pose Estimation", frame)
    cv2.destroyAllWindows()


def run_logic():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model.predict(frame)

        for result in results:
            if result.boxes.data.size(0) == 0:
                continue
            for i in range(result.boxes.data.size(0)):
                boxes = result.boxes.data[i]
                keypoints = result.keypoints.data[i]
                draw_keypoints_and_skeleton(frame, keypoints)
                fall_detected, bbox = fall_detection(boxes, keypoints)
                if fall_detected:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(
                        frame,
                        (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        color=(0, 0, 255),
                        thickness=5,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        "Person Fell down",
                        (11, 100),
                        0,
                        1,
                        [0, 0, 255],
                        thickness=3,
                        lineType=cv2.LINE_AA,
                    )
        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_ml()
