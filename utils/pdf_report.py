import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_pdf(type, current_time, image_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(base_dir, "laporan")  # Folder yang diinginkan
    os.makedirs(folder_path, exist_ok=True)
    
    # Tentukan nama file PDF dengan menambahkan folder path
    current_time = current_time.format("%H-%M-%S %d-%m-%Y")
    pdf_filename = os.path.join(folder_path, f"report_{type}_{current_time}.pdf")
    
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "LAPORAN DETEKSI FALL")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, f"Waktu dan Tanggal Kejadian: {current_time}")
    c.drawString(100, height - 120, "Tempat Kejadian: Tangga Teknik Fisika ITS")
    c.drawString(100, height - 140, f"Jenis Kejadian: {type.capitalize()}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 170, "Dokumentasi Kejadian:")

    try:
        image = PILImage.open(image_path)
        image = image.resize((640, 480))
        temp_image_path = "resized_image.jpg"
        image.save(temp_image_path)
        c.drawImage(temp_image_path, 100, height - 500)
        os.remove(temp_image_path)
    except Exception as e:
        print(f"Error saat memuat gambar: {e}")

    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.red)
    c.drawString(
        100, 100, "Terdeteksi Kejadan. Mohon segera diperiksa kondisi terkininya."
    )

    c.save()