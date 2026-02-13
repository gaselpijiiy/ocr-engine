from fastapi import FastAPI, UploadFile, File
import shutil
from app.ocr_service import extract_text

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "OCR API is running"}

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    temp_file = "temp.jpg"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(temp_file)

    return {"text": text}

# test main.py
