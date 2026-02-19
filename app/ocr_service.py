from paddleocr import PaddleOCR

# WAJIB ADA — ini yang bikin objek OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

def extract_text(image_path):

    result = ocr.ocr(image_path, cls=True)

    lines = []

    for line in result:
        if not line:
            continue

        for word in line:
            lines.append(word[1][0])

    return "\n".join(lines)
