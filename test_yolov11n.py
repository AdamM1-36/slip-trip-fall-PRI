from ultralytics import YOLO
import cv2
import math
from test_draw import draw_keypoints_and_skeleton


WEIGHT = 'yolo11n-pose.pt'
model = YOLO(WEIGHT)

# Path to the video file
video_path = "fall_dataset/videos/video_4.mp4"
cap = cv2.VideoCapture(video_path)
print(f"video fps is {cap.get(cv2.CAP_PROP_FPS)}")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_counter = 0

'''
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
'''

def check_fall_detection(
        x_min, y_min, x_max, y_max, 
        left_shoulder_y, left_foot_y, len_factor, 
        left_body_y, right_shoulder_y, right_foot_y, right_body_y):
    
    # Checking person for "horizontal" orientation
    dy = int(y_max) - int(y_min)
    dx = int(x_max) - int(x_min)
    difference = dy - dx

    if any(kp == 0 for kp in [left_shoulder_y, left_foot_y, len_factor, left_body_y, right_shoulder_y, right_foot_y, right_body_y]):
        print("Some keypoints are missing")
        return False
    
    # Calculate conditions, source: https://github.com/Y-B-Class-Projects/Human-Fall-Detection
    left_condition = (left_shoulder_y > left_foot_y - len_factor and
                      left_body_y > left_foot_y - (len_factor / 2) and
                      left_shoulder_y > left_body_y - (len_factor / 2))

    right_condition = (right_shoulder_y > right_foot_y - len_factor and
                       right_body_y > right_foot_y - (len_factor / 2) and
                       right_shoulder_y > right_body_y - (len_factor / 2))
    
    return left_condition or right_condition or difference < 0

def fall_detection(box, keypoint):
    box.detach().cpu().numpy()
    keypoint.detach().cpu().numpy()

    # Extract box dimension
    x_min, y_min = box[0], box[1]
    x_max, y_max = box[2], box[3]

    # Extract keypoints
    left_shoulder_y = keypoint[5][1]
    left_shoulder_x = keypoint[6][0]
    right_shoulder_y = keypoint[6][1]
    right_shoulder_x = keypoint[6][0]
    left_body_y = keypoint[11][1]
    left_body_x = keypoint[11][0]
    right_body_y = keypoint[12][1]
    right_body_x = keypoint[12][0]
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))

    left_knee_y = keypoint[13][1]
    left_knee_x = keypoint[13][0]
    right_knee_y = keypoint[14][1]
    right_knee_x = keypoint[14][0]

    left_foot_y = keypoint[15][1]
    right_foot_y = keypoint[16][1]

    fall_detected = check_fall_detection(x_min, y_min, x_max, y_max, left_shoulder_y, left_foot_y, len_factor, left_body_y, right_shoulder_y, right_foot_y, right_body_y)
    if fall_detected:
        return True, (x_min, y_min, x_max, y_max)
    return False, None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)

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
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                              thickness=5, lineType=cv2.LINE_AA)
                cv2.putText(frame, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)

    cv2.imshow("Pose Estimation", frame)
    frame_counter += 1
    print(frame_counter)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()