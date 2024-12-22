import asyncio
import math
import os
import queue
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import joblib
import numpy as np
import torch
from ultralytics import YOLO

# Test directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from .test_draw import draw_keypoints_and_skeleton
from utils.coco_keypoints import COCOKeypoints

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
warnings.simplefilter("ignore", UserWarning)

import tensorflow as tf

# CONFIG
# 3x speedup with tensorrt
USE_TENSORRT = True
# tf fall detection cause slowdown of about 12-14 ms per persoqn per frame
# -> fixed with batch processing (~15ms per frame)
USE_ML_FALL = True
FALL_THRESHOLD = 0.9  # Threshold very high?
# CONFIG END

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load models
if torch.cuda.is_available() and USE_TENSORRT:
    WEIGHT = "yolo11n-pose.engine"
    print("Using TensorRT fp16 model")
else:
    WEIGHT = "yolo11n-pose.pt"
    print("Using Pytorch model")
pose_model = YOLO(WEIGHT)
fall_model = tf.keras.models.load_model("ml/fall_detection_model.keras")

# Load video
video_path = "fall_dataset/videos/video_4.mp4"
cap = cv2.VideoCapture(video_path)
print(f"video fps is {cap.get(cv2.CAP_PROP_FPS)}")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# Thread safe queues
executor = ThreadPoolExecutor(max_workers=4)
frame_buffer = queue.Queue(maxsize=15)
processed_frame_buffer = queue.Queue(maxsize=15)
scaler = joblib.load("ml/scaler.pkl")


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
    dy = int(y_max) - int(y_min)
    dx = int(x_max) - int(x_min)
    difference = dy - dx

    if any(
        kp == 0
        for kp in [
            left_shoulder_y,
            left_foot_y,
            left_body_y,
            len_factor,
            right_shoulder_y,
            right_foot_y,
            right_body_y,
        ]
    ):
        print("Some keypoints are missing")
        return False

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


def fall_detection(boxes, keypoints, is_using_ml=False):
    keypoint_list = []
    box_coords = []
    fall_results = []

    for box, keypoint in zip(boxes, keypoints):
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

        keypoint_list.append(
            [
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
        )
        box_coords.append((x_min, y_min, x_max, y_max))

    keypoint_nparray = np.array(keypoint_list)
    keypoint_nparray = scaler.transform(keypoint_nparray)

    if is_using_ml:
        fall_predictions = fall_model.predict(keypoint_nparray)
        # print(f"Fall predictions: {fall_predictions}")
        for i, prediction in enumerate(fall_predictions):
            fall_detected = prediction[0] >= FALL_THRESHOLD
            x_min, y_min, x_max, y_max = box_coords[i]

            if fall_detected:
                fall_results.append((True, (x_min, y_min, x_max, y_max)))
            else:
                fall_results.append((False, None))
    else:
        for i, (x_min, y_min, x_max, y_max) in enumerate(box_coords):
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
                fall_results.append((True, (x_min, y_min, x_max, y_max)))
            else:
                fall_results.append((False, None))

    return fall_results


def fall_detection_worker(is_using_ml=True):
    while True:
        try:
            frame_data = frame_buffer.get(timeout=1)
        except queue.Empty:
            continue

        if frame_data is None:
            break

        frame, boxes, keypoints = frame_data
        if not boxes or not keypoints:
            processed_frame_buffer.put_nowait(frame)
            frame_buffer.task_done()
            continue

        try:
            fall_results = fall_detection(boxes, keypoints, is_using_ml)
            for fall_detected, bbox in fall_results:
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
        except Exception as e:
            print(f"Error in fall detection: {e}")

        try:
            processed_frame_buffer.put_nowait(frame)
        except queue.Full:
            print("W: Processed frame buffer full, skipping frame")

        frame_buffer.task_done()


async def process_frame(frame):
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        results = await loop.run_in_executor(executor, pose_model.predict, frame)

    boxes, keypoints = [], []
    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue
        output_size = len(result.boxes.data)
        if output_size == 0:
            continue
        for i in range(output_size):
            boxes.append(result.boxes.data[i])
            keypoints.append(result.keypoints.data[i])
            draw_keypoints_and_skeleton(frame, keypoints[-1])

    if boxes and keypoints:
        try:
            frame_buffer.put_nowait((frame.copy(), boxes, keypoints))
            print(f"I: Frame counter: {frame_buffer.qsize()}")
        except queue.Full:
            print("W: Frame buffer full, skipping frame")
    else:
        print("W: No person detected in frame")


async def run(is_using_ml=True):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        await process_frame(frame)
        if not processed_frame_buffer.empty():
            processed_frame = processed_frame_buffer.get_nowait()
            cv2.imshow(
                "ML Processing" if is_using_ml else "Non-ML Processing", processed_frame
            )

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    frame_buffer.put(None)


if __name__ == "__main__":
    fall_thread = threading.Thread(
        target=fall_detection_worker, args=(USE_ML_FALL,), daemon=True
    )
    fall_thread.start()
    asyncio.run(run(USE_ML_FALL))
