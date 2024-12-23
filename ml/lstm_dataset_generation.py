import os
import sys

import cv2
import pandas as pd
import importlib.util
from ultralytics import YOLO

parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(parent_dir)
utils_dir = os.path.join(parent_dir, 'utils')


# Import using absolute path
coco_keypoints_path = os.path.join(utils_dir, 'coco_keypoints.py')
spec = importlib.util.spec_from_file_location("coco_keypoints", coco_keypoints_path)
coco_keypoints = importlib.util.module_from_spec(spec)
sys.modules["coco_keypoints"] = coco_keypoints
spec.loader.exec_module(coco_keypoints)
COCOKeypoints = coco_keypoints.COCOKeypoints
extract_keypoints = coco_keypoints.extract_keypoints

model = YOLO("ml/yolo11m-pose.engine")

base_folder = os.path.join(os.path.dirname(__file__), "../dataset_primer/edited/")
print(base_folder)
target_size = (640, 640)
labels = ["Slip", "Trip", "Walking"]
label_mapping = {
    "Slip": "Slip",
    "SlipFall": "Slip",
    "Trip": "Trip",
    "TripFall": "Trip",
    "Walking": "Walking"
}

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


MAX_TIMESTEPS = 24
MIN_TIMESTEPS = 12

def process_video(video_path, model, show_output=False):
    cap = cv2.VideoCapture(video_path)
    data = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = 60
    print(f"total : max frames = {total_frames} : {max_frames}")
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, target_size)

        results = model.track(frame, persist=True, verbose=False)
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
                        
                        kp.frame_index = frame_count
                        data.append(kp)
                        frame_count += 1

                        # Keep at most the last MAX_TIMESTEPS data
                        if len(data) > MAX_TIMESTEPS:
                            data = data[-MAX_TIMESTEPS:]

        if show_output:
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
    cap.release()
    if show_output:
        cv2.destroyAllWindows()
    return data

def create_dataset_with_cases(base_folder, model, show_output=False):
    data = []
    dropped_videos_count = {label: {"upstairs": 0, "downstairs": 0} for label in labels}

    cases = ["upstairs", "downstairs"]
    segment_index = 0  # Initialize segment index

    for case in cases:
        case_folder = os.path.join(base_folder, case)
        if not os.path.isdir(case_folder):
            continue

        for original_label in label_mapping.keys():
            label_path = os.path.join(case_folder, original_label)
            if os.path.isdir(label_path):
                print(f"Processing folder: {case}/{original_label}")
                video_files = sorted(
                    [f for f in os.listdir(label_path) if f.endswith(".mp4")]
                )
                for video_file in video_files:
                    video_path = os.path.join(label_path, video_file)
                    personalize_id = original_label == "Walking"
                    frames = process_video(video_path, model, show_output)

                    if len(frames) >= MIN_TIMESTEPS:
                        print(f"Processing video: {video_file}")
                        for frame in frames:
                            frame_data = {
                                k: v
                                for k, v in frame.__dict__.items()
                                if k in COCO_KEYPOINTS_NO_CONF
                            }
                            frame_data["segment_index"] = segment_index
                            frame_data["person_index"] = frame.person_index
                            if not personalize_id:
                                frame_data["person_index"] = 0
                            frame_data["frame_index"] = frame.frame_index  # Frame index starts at 1
                            frame_data["label"] = label_mapping[original_label]
                            frame_data["case"] = case
                            data.append(frame_data)
                        segment_index += 1  # Increment segment index for the next video
                    else:
                        dropped_videos_count[label_mapping[original_label]][case] += 1

    columns = COCO_KEYPOINTS_NO_CONF + ["segment_index", "person_index", "frame_index", "label", "case"]
    df = pd.DataFrame(data, columns=columns)

    # Print the dropped videos count for each label and case
    for label, case_counts in dropped_videos_count.items():
        for case, count in case_counts.items():
            print(f"Number of videos dropped for {label} ({case}): {count}")

    return df

if __name__ == "__main__":
    show_output = False  # Set this to False if you don't want to show the output
    # dataset = create_dataset(base_folder, model, show_output)
    dataset = create_dataset_with_cases(base_folder, model, show_output)
    dataset.to_csv(
        os.path.join(os.path.dirname(__file__), "../dataset_primer/edited/lstm_dataset_with_cases.csv"),
        index=False,
    )
    print("Dataset saved to: ../dataset_primer/edited/lstm_dataset_with_cases.csv")
