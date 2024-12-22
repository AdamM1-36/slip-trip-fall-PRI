import os
from threading import Thread

import cv2
from openpyxl.drawing.image import Image

from .pdf_report import generate_pdf
from .telegram_report import send_telegram_message, send_telegram_photo


def process_report(wb, ws, frame, type, start_time, current_time, token, chat_id, row=0):
    col_time = "A"
    col_type = "B"
    col_image = "C"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(base_dir, f"pri_images/{type}")
    os.makedirs(folder_path, exist_ok=True)

    pict = f"{folder_path}/{current_time}.jpg"
    cv2.imwrite(pict, frame)

    img = Image(pict)
    img.height = 300
    img.width = 400
    
    row += 20
    ws[f"{col_time}{row}"] = current_time
    ws[f"{col_type}{row}"] = type
    ws.add_image(img, f"{col_image}{row}")
    os.makedirs("excel", exist_ok=True)
    wb.save(f"excel/result_{start_time}.xlsx")
    generate_pdf(type, current_time, pict)
    
    message = f"Terdeteksi kejadian {type.capitalize()} pada {current_time}. Mohon segera diperiksa kondisi terkininya"
    Thread(target=send_telegram_message, args=(message, token, chat_id)).start()
    Thread(target=send_telegram_photo, args=(pict, token, chat_id)).start()