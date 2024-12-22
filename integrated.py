import asyncio
import os
import sys
import time
from asyncio import Queue
from datetime import datetime
from threading import Lock, Thread

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import pad_sequences
from openpyxl import Workbook
from starlette.websockets import WebSocket, WebSocketDisconnect
from ultralytics import YOLO

from utils.coco_keypoints import COCOKeypoints, extract_keypoints
from utils.report import process_report
from utils.telegram_report import listen_to_tele_bot

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")

gpus = tf.config.experimental.list_physical_devices("GPU")
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

# Global variables
load_dotenv()
cap = cv2.VideoCapture(os.getenv("RTSP_URL"))
cap_lock = Lock()
MIN_FRAMES = 20
MAX_TIMESTEPS = 30
rolling_buffer = {}
event_queue = Queue()

# Load models and scalers
KEYPOINT_MODEL_PATH = "ml/yolo11n-pose.engine"
LSTM_MODEL_PATH = "ml/lstm/slip_fall_detector_V2_with-cases.keras"
FALL_MODEL_PATH = "ml/fall.engine"
UPSTAIRS = True

if UPSTAIRS:
    SCALER_PATH = "ml/lstm/scaler_upstairs.pkl"
else:
    SCALER_PATH = "ml/lstm/scaler_downstairs.pkl"

lstm_model = load_model(LSTM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
keypoint_model = YOLO(KEYPOINT_MODEL_PATH, task="pose")
fall_model = YOLO(FALL_MODEL_PATH, task="detect")


# Konfigurasi bot Telegram
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
status_detection = True  # Status deteksi awal

# Konfigurasi excel report
wb = Workbook()
ws = wb.active

# Waktu mulai program
start_time = datetime.now().strftime("%H-%M-%S %d-%m-%Y")

target_size = (640, 480)
# labels = ["Slip", "SlipFall", "Trip", "TripFall", "Walking"]
labels = ["Slip", "Trip", "Walking", "Fall"]

# Define columns without confidence values
COCO_KEYPOINTS_NO_CONF = COCOKeypoints.keypoints_no_conf()

Thread(target=listen_to_tele_bot, args= (TOKEN,), daemon=True).start()

def report_detection(frame, detection_type, start_time, current_time, token, chat_id, wb, ws, websocket, state):
    thread = Thread(target=process_report, args=(wb, ws, frame, detection_type, start_time, current_time, token, chat_id))
    thread.start()
    asyncio.create_task(send_alert_async(websocket, state.lower()))
    print(f"Reported detection: {detection_type}")

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

def lstm_predict(results, tracker_id, lstm_model, scaler):
    global rolling_buffer

    if tracker_id not in rolling_buffer:
        rolling_buffer[tracker_id] = []
        print(f"New person added to rolling_buffer with tracker_id: {tracker_id}")
        
    keypoints = extract_keypoints([results])
    if keypoints:
        for keypoint in keypoints:
            if keypoint.person_index == tracker_id:
                keypoint_coord = [getattr(keypoint, k, None) for k in COCO_KEYPOINTS_NO_CONF]

                if filter_keypoints(np.array(keypoint_coord)):
                    rolling_buffer[tracker_id].append(keypoint_coord)
                    rolling_buffer[tracker_id] = rolling_buffer[tracker_id][-MAX_TIMESTEPS:]
                    print(f"Updated rolling_buffer for tracker_id {tracker_id}")

    # Ensure we have enough frames before making a prediction
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
        print(f"Making prediction for tracker_id {tracker_id}")
        prediction = lstm_model.predict(frame_data, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class: {predicted_class} : {labels[predicted_class]}")
        return labels[predicted_class]

    return "Unknown"  # No prediction if insufficient frames

def lstm_results_loop(keypoint_results, person_labels, frame):
    for keypoint_result in keypoint_results:
        for obj in keypoint_result.boxes:
            tracker_id = obj.id
            if tracker_id is not None:
                tracker_id = tracker_id.item()
            else:
                continue

            label = lstm_predict(keypoint_result, tracker_id, lstm_model, scaler)
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
    return frame, person_labels

async def get_frame_async():
    global cap
    with cap_lock:
        ret, frame = await asyncio.to_thread(cap.read)
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        return None
    return frame

async def send_alert_async(websocket, payload):
    print("Pesan terkirim ke ESP32")
    await websocket.send_text(payload)

def run_fall_detection(frame, results):
    results['fall_results'] = fall_model(frame, conf=0.8, verbose=False)
    
def run_keypoint_and_lstm(frame, person_labels, results):
    keypoint_results = keypoint_model.track(frame, persist=True, verbose=False, conf=0.5)
    frame, person_labels = lstm_results_loop(keypoint_results, person_labels, frame)
    results['frame'] = frame
    results['person_labels'] = person_labels

async def app(scope, receive, send):
    websocket = WebSocket(scope=scope, receive=receive, send=send)
    await websocket.accept()
    await asyncio.sleep(3)
    
    try:
        save_output = False
        show_output = True
        person_labels = {}
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Frame width: {frame_width}, Frame height: {frame_height}, FPS: {fps}")
        
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_folder = os.path.join(os.getcwd(), "lstm_output")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(
                output_folder, os.path.basename('output.mp4')
            )
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        last_detection_time = 0  
        detection_delay = 10  # Delay waktu dalam detik
        fall_was_detected = False
        prev_frame_time = time.time()
        fall_frame = 0
        
        cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Output Video", 1280, 720)
        while cap.isOpened():
            now = datetime.now()
            current_time = now.strftime("%H-%M-%S %d-%m-%Y")
            frame = await get_frame_async()
            if frame is None:
                break
            # FPS counter
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Tampilkan status deteksi pada frame
            status_text = "Deteksi: ON" if status_detection and not fall_was_detected else "Deteksi: OFF"
            cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            if status_detection:
                results = {}
                
                # Create threads
                fall_thread = Thread(target=run_fall_detection, args=(frame, results), daemon=True)
                keypoint_lstm_thread = Thread(target=run_keypoint_and_lstm, args=(frame, person_labels, results), daemon=True)
                fall_thread.start()
                keypoint_lstm_thread.start()
                fall_thread.join()
                keypoint_lstm_thread.join()
                
                fall_results = results['fall_results']
                frame, person_labels = results['person_labels']

                for fall_result in fall_results:
                    class_id_list = list(fall_result.boxes.cls)
                    # print(f"fall frame: {fall_frame}")
                    if 0 in class_id_list:
                        fall_frame += 1 
                    else:
                        # test
                        fall_frame = max(0, fall_frame - 5)
                        # fall_frame = 0
                        
                    if fall_frame == 20 and not fall_was_detected:
                        print("Detected state: Fall")
                        frame = fall_result.plot()
                        report_detection(frame, 'fall', start_time, current_time, TOKEN, CHAT_ID, wb, ws, websocket, 'fall')
                        last_detection_time = time.time()
                        fall_frame = 0
                        fall_was_detected = True
                        break
                
                if not fall_was_detected:
                    for state in person_labels.values():
                        if state != "Unknown" and state != "Walking":
                            print(f"Detected state: {state}")
                            report_detection(frame, state.lower(), start_time, current_time, TOKEN, CHAT_ID, wb, ws, websocket, state)
                            last_detection_time = time.time()
                            rolling_buffer.clear()
                            print("Rolling_buffer cleared")
                            fall_was_detected = True
                            break
                    person_labels.clear()

            if fall_was_detected and (time.time() - last_detection_time) >= detection_delay:
                print("Turning detection back ON")
                fall_was_detected = False
                last_detection_time = 0

            if show_output and isinstance(frame, np.ndarray):
                cv2.imshow("Output Video", frame)
            if save_output and isinstance(frame, np.ndarray):
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
    except WebSocketDisconnect:
        print("WebSocket connection was disconnected.")
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        await websocket.close()
        