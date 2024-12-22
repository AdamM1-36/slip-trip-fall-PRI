import os

import cv2
from ultralytics import YOLO

model = YOLO("../yolo11n-pose.pt")

video_folder = os.path.join(os.path.dirname(__file__), "../dataset_primer/trip/")
target_size = (640, 480)


def process_video(video_folder, model):
    cap = cv2.VideoCapture(video_folder)

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
                print(tracker_id)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO11 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for root, dirs, files in os.walk(video_folder):
        files.sort()
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                print(f"video_path: {video_path}")
                process_video(video_path, model)
