from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='id')
result = ocr.ocr("D:\\ocr-engine\\data\\test.jpg")

print("===== HASIL OCR BERSIH =====")

texts = []

for res in result:
    for line in res:
        text = line[1][0]
        texts.append(text)

final_text = "\n".join(texts)
print(final_text)
