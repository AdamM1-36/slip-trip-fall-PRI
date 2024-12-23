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
from utils.shared_state import shared_state

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
# cap = cv2.VideoCapture("final.mp4")
# cap = cv2.VideoCapture("dataset_primer/edited/upstairs/Walking/clip_Walking0041_003.mp4")

cap_lock = Lock()
MIN_FRAMES = 16
MAX_TIMESTEPS = 24
rolling_buffer = {}
event_queue = Queue()

# Load models and scalers
KEYPOINT_MODEL_PATH = "models_used/yolo11m-pose.engine"
FALL_MODEL_PATH = "models_used/fall.engine"
UPSTAIRS = True
REPORT = True

if UPSTAIRS:
    LSTM_MODEL_PATH = "ml/lstm/slip_fall_detector_V3_upstairs.keras"
    SCALER_PATH = "ml/lstm/scaler_upstairs.pkl"
else:
    SCALER_PATH = "ml/lstm/scaler_downstairs.pkl"

lstm_model = load_model(LSTM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
keypoint_model = YOLO(KEYPOINT_MODEL_PATH, task="pose")
fall_model = YOLO(FALL_MODEL_PATH, task="detect")
results_lock = Lock()

# Konfigurasi bot Telegram
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
shared_state.status_detection = True  # Status deteksi awal

# Konfigurasi excel report
wb = Workbook()
ws = wb.active

# Waktu mulai program
start_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")

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
                    # "left_hip_x",
                    # "left_hip_y",
                    # "right_hip_x",
                    # "right_hip_y",
                    # "left_knee_x",
                    # "left_knee_y",
                    # "right_knee_x",
                    # "right_knee_y",
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
                    # print(f"Appended rolling_buffer for tracker_id {tracker_id}")
                    if len(rolling_buffer[tracker_id]) > MAX_TIMESTEPS:
                        rolling_buffer[tracker_id] = rolling_buffer[tracker_id][-MAX_TIMESTEPS:]
                        # print(f"dropped frames for tracker_id {tracker_id}")
                    
                    # print(f"Length of rolling_buffer for tracker_id {tracker_id}: {len(rolling_buffer[tracker_id])}")

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
        )

        # Predict the class
        print(f"Making prediction for tracker_id {tracker_id}")
        prediction = lstm_model.predict(frame_data, verbose=False)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class: {predicted_class} : {labels[predicted_class]}")
        return labels[predicted_class]

    return "Unknown"  # No prediction if insufficient frames

def lstm_keypoint_classification(keypoint_results, person_labels, frame):
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

### FALL DETECTION FUNCTION START ###

def run_fall_detection(frame, results):
    with results_lock:
        results['fall_results'] = fall_model(frame, conf=0.8, verbose=False)
        
def handle_fall_detection(fall_was_detected, frame, fall_results, fall_frame_counter, last_detection_time, websocket):
    for fall_result in fall_results:
        class_id_list = list(fall_result.boxes.cls)
        if 0 in class_id_list:
            fall_frame_counter += 1 
        else:
            # test
            fall_frame_counter = max(0, fall_frame_counter - 5)
            # fall_frame_counter = 0
            
        if fall_frame_counter == 20 and not fall_was_detected:
            print("Detected state: Fall")
            frame = fall_result.plot()
            current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
            if REPORT:
                report_detection(frame, 'fall', start_time, current_time, TOKEN, CHAT_ID, wb, ws, websocket, 'fall')
            last_detection_time = time.time()
            fall_frame_counter = 0
            fall_was_detected = True
            break
    # print(F"fall_frame_counter: {fall_frame_counter}")
    return fall_was_detected, fall_frame_counter, frame, last_detection_time
    
### FALL DETECTION FUNCTION END ###

### LSTM DETECTION FUNCTION START ###

def run_keypoint_and_lstm(frame, person_labels, results):
    keypoint_results = keypoint_model.track(frame, persist=True, verbose=False, conf=0.6)
    frame, person_labels = lstm_keypoint_classification(keypoint_results, person_labels, frame)
    with results_lock:
        results['frame'] = frame
        results['person_labels'] = person_labels
        
def handle_lstm_detection(lstm_was_detected, frame, person_labels, lstm_frame_counter, last_detection_time, last_detected_state, websocket):
    global rolling_buffer
    for tracker_id, state in person_labels.items():
        if state in ["Slip", "Trip"]:
            lstm_frame_counter[state] += 1
            last_detected_state[tracker_id] = state
            # print lstm_frame_counter dictionary key and values:
            for key, value in lstm_frame_counter.items():
                print(f"{key}: {value}")
        else:
            # print(f"Reset logic")
            if tracker_id in last_detected_state and last_detected_state[tracker_id] in lstm_frame_counter:
                lstm_frame_counter[last_detected_state[tracker_id]] = max(0, lstm_frame_counter[last_detected_state[tracker_id]] - 3)
        
        if lstm_frame_counter[state] == 5 and not lstm_was_detected:
            print(f"Detected state: {state}")
            current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
            if REPORT:
                report_detection(frame, state.lower(), start_time, current_time, TOKEN, CHAT_ID, wb, ws, websocket, state)
            last_detection_time = time.time()
            lstm_was_detected = True
            break
    
    if lstm_was_detected:
        rolling_buffer.clear()
        person_labels.clear()
        lstm_frame_counter = {key: 0 for key in lstm_frame_counter.keys()}
        print("LSTM buffer cleared")
    
    return lstm_was_detected, lstm_frame_counter, frame, last_detection_time, last_detected_state

### LSTM DETECTION FUNCTION END ###

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

        last_fall_time = 0
        last_lstm_time = 0
        detection_delay = 15  # Delay jeda antar deteksi dalam detik
        fall_was_detected = False
        lstm_was_detected = False
        prev_frame_time = time.time()
        fall_frame_counter = 0
        lstm_frame_counter = {label: 0 for label in ['Slip', 'Trip', 'Walking', 'Unknown']}
        lstm_last_detected_state = {}
        
        cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Output Video", 1280, 720)
        while cap.isOpened():
            frame = await get_frame_async()
            if frame is None:
                break
            
            # FPS counter
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # Show status text
            status_text = "Deteksi: ON" if shared_state.status_detection and not fall_was_detected else "Deteksi: OFF"
            cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            if shared_state.status_detection:
                # Results dictionary to store results from threads
                results = {}
                
                # Do detection in parallel using threads  
                # if not fall_was_detected:
                if not fall_was_detected:
                    # print("Running fall detection")
                    fall_thread = Thread(target=run_fall_detection, args=(frame, results), daemon=True)
                    fall_thread.start()
                    fall_thread.join()
                    if 'fall_results' not in results :
                        print("Threads failed to access fall results correctly")
                        continue  # Skip processing for this frame
                    fall_results = results['fall_results']
                    
                    # Handle fall detection and report
                    fall_was_detected, fall_frame_counter, frame, last_fall_time = \
                    handle_fall_detection(fall_was_detected, frame, fall_results, fall_frame_counter, last_fall_time, websocket)
                
                if not lstm_was_detected:
                    # print("Running LSTM detection")
                    keypoint_lstm_thread = Thread(target=run_keypoint_and_lstm, args=(frame, person_labels, results), daemon=True)
                    keypoint_lstm_thread.start()
                    keypoint_lstm_thread.join()
                    
                    if 'person_labels' not in results or 'frame' not in results:
                        print("Threads failed to access lstm results correctly")
                        continue
                    # Access lstm results from threads variable
                    frame, person_labels = results['frame'], results['person_labels']
                    
                    # Handle LSTM detection and reporXt
                    lstm_was_detected, lstm_frame_counter, frame, last_lstm_time, lstm_last_detected_state = \
                        handle_lstm_detection(lstm_was_detected, frame, person_labels, lstm_frame_counter, last_lstm_time, lstm_last_detected_state, websocket)
                
                # Handle detection system pause and reset
                if fall_was_detected and (time.time() - last_fall_time) >= detection_delay:
                    print("Turning detection back ON")
                    fall_was_detected = False
                    last_fall_time = 0
                
                if lstm_was_detected and (time.time() - last_lstm_time) >= detection_delay:
                    print("Turning detection back ON")
                    lstm_was_detected = False
                    last_lstm_time = 0

            # Handle output display and saving
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
