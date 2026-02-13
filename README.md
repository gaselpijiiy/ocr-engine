# OCR API

## Menjalankan server
1. Install dependency
pip install -r requirements.txt

2. Jalankan server
uvicorn app.main:app --reload

## Endpoint
POST /ocr

Form-data:
file: image
