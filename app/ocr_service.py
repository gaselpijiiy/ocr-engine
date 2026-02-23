import os
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# Load model sekali saja
ocr = PaddleOCR(use_angle_cls=True, lang="en")


def extract_text_from_image(image_path):
    result = ocr.ocr(image_path)
    text = ""

    if result and result[0]:
        for line in result[0]:
            text += line[1][0] + " "

    return text.strip()


def extract_text(file_path):

    ext = file_path.lower().split(".")[-1]

    # ✅ IMAGE
    if ext in ["jpg", "jpeg", "png"]:
        return extract_text_from_image(file_path)

    # ✅ PDF (SCAN)
    elif ext == "pdf":

        images = convert_from_path(
            file_path,
            poppler_path=r"D:\ocr-engine\poppler-25.12.0\Library\bin"
        )

        full_text = []

        for i, image in enumerate(images):
            temp_img = f"temp_page_{i}.jpg"
            image.save(temp_img, "JPEG")

            text = extract_text_from_image(temp_img)
            full_text.append(text)

            os.remove(temp_img)

        return "\n\n".join(full_text)

    else:
        return "Unsupported file type"