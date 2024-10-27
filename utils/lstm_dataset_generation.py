from ultralytics import YOLO
import cv2
import os
import pandas as pd
from coco_keypoints import extract_keypoints

show_output = True

model = YOLO(os.path.join(os.path.dirname(__file__), "../yolo11n-pose.pt"))
video_folder = os.path.join(
    os.path.dirname(__file__), "../dataset_primer/2/Trip/Labels"
)

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


def process_video(video_path, model, show_output=False):
    cap = cv2.VideoCapture(video_path)
    event_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        keypoints = extract_keypoints(results)
        if keypoints:
            event_frames.extend(keypoints)
            # Keep only the last 30 frames
            event_frames = event_frames[-30:]
        if show_output:
            # Use result.plot() to visualize the results
            for result in results:
                plotted_frame = result.plot()
                cv2.imshow("Frame", plotted_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cap.release()
    if show_output:
        cv2.destroyAllWindows()
    return event_frames


def create_dataset(video_folder, model, show_output=False):
    data = []
    for i in range(1, 10):
        video_path = os.path.join(video_folder, f"Trip{i}.mp4")
        frames = process_video(video_path, model, show_output)
        if len(frames) >= 30:
            for frame in frames:
                frame_data = {
                    k: v
                    for k, v in frame.__dict__.items()
                    if k in COCO_KEYPOINTS_NO_CONF
                }
                frame_data["segment_index"] = i
                data.append(frame_data)
    # Create DataFrame with the new segment_index column
    columns = COCO_KEYPOINTS_NO_CONF + ["segment_index"]
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    dataset = create_dataset(video_folder, model, show_output)
    dataset.to_csv(
        os.path.join(
            os.path.dirname(__file__), "../dataset_primer/2/Trip/trip_dataset.csv"
        ),
        index=False,
    )
