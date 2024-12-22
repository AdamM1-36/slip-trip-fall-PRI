import time

import requests


def send_telegram_message(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=data)
        # print(f"data telegram : {data}")
        if response.status_code == 200:
            print("Peringatan terkirim ke Telegram.")
        else:
            print(f"Gagal mengirim pesan. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error saat mengirim pesan: {e}")

def send_telegram_photo(photo_path, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {"photo": open(photo_path, "rb")}
    data = {"chat_id": chat_id}
    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("Foto terkirim ke Telegram.")
        else:
            print(f"Gagal mengirim foto. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error saat mengirim foto: {e}")

def listen_to_tele_bot(token):
    global status_detection
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    last_update_id = None  # Variabel untuk menyimpan ID pembaruan terakhir yang sudah diproses

    while True:
        try:
            # Membuat URL dengan offset agar hanya mengambil pembaruan yang lebih baru dari last_update_id
            if last_update_id:
                url_with_offset = f"{url}?offset={last_update_id + 1}"
            else:
                url_with_offset = url

            response = requests.get(url_with_offset)
            
            if response.status_code == 200:
                updates = response.json().get("result", [])
                if updates:
                    for update in updates:
                        update_id = update["update_id"]
                        message = update.get("message", {}).get("text", "").lower()

                        # Memproses hanya jika update_id lebih besar dari yang terakhir diproses
                        if last_update_id is None or update_id > last_update_id:
                            last_update_id = update_id  # Perbarui last_update_id
                            if message == "matikan":
                                status_detection = False
                                send_telegram_message("Sistem deteksi telah dimatikan.")
                            elif message == "nyalakan":
                                status_detection = True
                                send_telegram_message("Sistem deteksi telah dihidupkan.")

        except Exception as e:
            print(f"Error saat mendengarkan bot: {e}")

        # Menunggu beberapa detik sebelum mengambil pembaruan selanjutnya
        time.sleep(2)
