import cv2
from ultralytics import YOLO
from dotenv import load_dotenv


model = YOLO("trained_models/v5-new_64_50.engine", task="detect")
# model = YOLO("trained_models/v5-new_64_50.pt")
cap = cv2.VideoCapture(os.getenv("RTSP_URL"))
# cap = cv2.VideoCapture('fall_dataset/videos/video_5.mp4')
# cap = cv2.VideoCapture("new_datasets/3.mp4")
while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame, conf=0.8, imgsz=640)
        for result in results:
       
            frame = result.plot()
            cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()