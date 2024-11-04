import cv2
import os
from ultralytics import YOLO

model = YOLO("../yolo11n-pose.pt")

video_path = os.path.join(os.path.dirname(__file__), "../dataset_primer/2/1_edited.mp4")
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
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
    else:
        break

cap.release()
cv2.destroyAllWindows()
