from paddleocr import PaddleOCR

# load model sekali saat worker start
ocr = PaddleOCR(use_angle_cls=True, lang="en")

def extract_text(image_path):
    result = ocr.ocr(image_path)
    text = ""

    for line in result[0]:
        text += line[1][0] + " "

    return text
