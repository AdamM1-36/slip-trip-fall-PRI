import cv2
import sys
import os
import pandas as pd
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils.coco_keypoints import extract_keypoints, COCOKeypoints

model = YOLO("../yolo11n-pose.pt")

base_folder = os.path.join(os.path.dirname(__file__), "../dataset_primer/")
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


def process_video(video_path, model, keep_all_frames=False, show_output=False):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, target_size)

        results = model.track(frame, persist=True)
        for result in results:
            for obj in result.boxes:
                tracker_id = obj.id
                if tracker_id is not None:
                    tracker_id = tracker_id.numpy()
                else:
                    continue

                keypoints = extract_keypoints([result])
                if keypoints:
                    for kp in keypoints:
                        kp.person_index = int(tracker_id)
                        frames.append(kp)
                        # Keep at most the last 30 frames for labels except Walking
                        if not keep_all_frames:
                            frames = frames[-30:]

        if show_output:
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_output:
        cv2.destroyAllWindows()
    return frames


def create_dataset(base_folder, model, show_output=False):
    data = []
    for label_folder in os.listdir(base_folder):
        label_path = os.path.join(base_folder, label_folder)
        if os.path.isdir(label_path):
            print(f"Processing folder: {label_folder}")
            video_files = sorted(
                [f for f in os.listdir(label_path) if f.endswith(".mp4")]
            )
            for i, video_file in enumerate(video_files):
                video_path = os.path.join(label_path, video_file)
                keep_all_frames = label_folder == "Walking"
                frames = process_video(video_path, model, keep_all_frames, show_output)
                if len(frames) >= 10:
                    print(f"Processed video: {video_file}")
                    for frame in frames:
                        frame_data = {
                            k: v
                            for k, v in frame.__dict__.items()
                            if k in COCO_KEYPOINTS_NO_CONF
                        }
                        frame_data["segment_index"] = i
                        frame_data["person_index"] = frame.person_index
                        frame_data["label"] = label_folder
                        data.append(frame_data)
    data.sort(key=lambda x: (x["segment_index"], x["person_index"]))
    columns = COCO_KEYPOINTS_NO_CONF + ["segment_index", "person_index", "label"]
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    show_output = False  # Set this to False if you don't want to show the output
    dataset = create_dataset(base_folder, model, show_output)
    dataset.to_csv(
        os.path.join(os.path.dirname(__file__), "../dataset_primer/lstm_dataset.csv"),
        index=False,
    )
    print("Dataset saved to: ../dataset_primer/lstm_dataset.csv")
