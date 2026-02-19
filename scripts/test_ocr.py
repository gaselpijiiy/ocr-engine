from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

image_path = "data/test1.jpg"   

result = ocr.ocr(image_path, cls=True)

print(result)  
