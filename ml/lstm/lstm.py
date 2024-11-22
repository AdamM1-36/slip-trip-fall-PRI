import os
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import pad_sequences
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from utils.coco_keypoints import COCOKeypoints, extract_keypoints

gpus = tf.config.experimental.list_physical_devices("GPU")
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

# Global variables
MIN_FRAMES = 10
MAX_TIMESTEPS = 30
rolling_buffer = {}

# Load models and scalers
model_path = "ml/lstm/slip_fall_detector.keras"
scaler_path = "ml/lstm/lstm_scaler.pkl"
lstm_model = load_model(model_path)
scaler = joblib.load(scaler_path)
yolo_model = YOLO("yolo11n-pose.engine")

base_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../dataset_primer/"
)
target_size = (640, 480)
labels = ["Slip", "SlipFall", "Trip", "TripFall", "Walking"]

# Define columns without confidence values
COCO_KEYPOINTS_NO_CONF = [
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


def filter_keypoints(keypoints):
    if np.all(keypoints == 0):
        return False
    if np.any(
        keypoints[
            [
                COCO_KEYPOINTS_NO_CONF.index(k)
                for k in [
                    "left_shoulder_x",
                    "left_shoulder_y",
                    "right_shoulder_x",
                    "right_shoulder_y",
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
            ]
        ]
        == 0
    ):
        return False
    return True


def process_frame(results, tracker_id, lstm_model, scaler):
    global rolling_buffer

    if tracker_id not in rolling_buffer:
        rolling_buffer[tracker_id] = []  # Initialize buffer for a new person

    keypoints = extract_keypoints([results])
    if keypoints:
        for kp in keypoints:
            if kp.person_index == tracker_id:
                keypoint_coord = []
                for keypoint_name in COCO_KEYPOINTS_NO_CONF:
                    value = getattr(kp, keypoint_name, None)
                    keypoint_coord.append(value)

                if filter_keypoints(np.array(keypoint_coord)):
                    rolling_buffer[tracker_id].append(keypoint_coord)
                    rolling_buffer[tracker_id] = rolling_buffer[tracker_id][
                        -MAX_TIMESTEPS:
                    ]

    print(f"Tracker ID: {tracker_id}, Frames: {len(rolling_buffer[tracker_id])}")
    # Predict only if we have at least `min_frames` in the buffer
    if len(rolling_buffer[tracker_id]) >= MIN_FRAMES:
        frame_data = np.array(rolling_buffer[tracker_id])
        frame_data = frame_data.reshape(-1, len(COCO_KEYPOINTS_NO_CONF))
        frame_data_df = pd.DataFrame(frame_data, columns=COCO_KEYPOINTS_NO_CONF)
        frame_data = scaler.transform(frame_data_df)

        # Pad or truncate to max_timesteps as required
        frame_data = pad_sequences(
            [frame_data],
            maxlen=MAX_TIMESTEPS,
            dtype="float32",
            padding="post",
            truncating="post",
        )

        # Predict the class
        prediction = lstm_model.predict(frame_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return labels[predicted_class]

    return "Unknown"  # No prediction if insufficient frames


def process_video(
    video_path, yolo_model, lstm_model, scaler, show_output=True, save_output=False
):
    cap = cv2.VideoCapture(video_path)
    person_labels = {}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_folder = os.path.join(os.getcwd(), "lstm_output")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(
            output_folder, os.path.basename(video_path).replace(".mp4", "_output.mp4")
        )
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, target_size)

        results = yolo_model.track(frame, persist=True)
        for result in results:
            for obj in result.boxes:
                tracker_id = obj.id
                if tracker_id is not None:
                    tracker_id = tracker_id.item()
                else:
                    continue

                label = process_frame(result, tracker_id, lstm_model, scaler)
                if label:
                    person_labels[tracker_id] = label

                    # Draw bounding box and label
                    bbox = obj.xyxy[0].numpy().astype(int)
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (bbox[0], bbox[3] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

        if show_output and isinstance(frame, np.ndarray):
            cv2.imshow("YOLO11 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if save_output and isinstance(frame, np.ndarray):
            out.write(frame)

    cap.release()
    if show_output:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_folder = os.path.join(base_folder, "Walking")
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing video: {video_file}")
        process_video(
            video_path,
            yolo_model,
            lstm_model,
            scaler,
            show_output=True,
            save_output=True,
        )
        print("Not enough frames for prediction.")

        # Pause between videos
        input("Press Enter to continue to the next video...")
