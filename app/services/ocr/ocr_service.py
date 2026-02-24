import time
from paddleocr import PaddleOCR


class OCRService:

    def __init__(self):
        """
        Smart Routing OCRService untuk global multi-language
        PaddleOCR, confidence-based, backward compatible
        """
        # Load models
        self.models = {
            "latin": PaddleOCR(use_angle_cls=True, lang="latin"),
            "japan": PaddleOCR(use_angle_cls=True, lang="japan"),
            "ch": PaddleOCR(use_angle_cls=True, lang="ch")
        }

        # Threshold confidence
        self.LATIN_THRESHOLD = 0.80

    def _run_model(self, image_path, model_key):
        start = time.time()

        # OCR
        result = self.models[model_key].ocr(image_path, cls=True)

        texts = []
        confs = []

        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])
                confs.append(line[1][1])

        avg_conf = sum(confs) / len(confs) if confs else 0

        return {
            "text": " ".join(texts),
            "confidence": round(avg_conf, 3),
            "model": model_key,
            "processing_time_ms": int((time.time() - start) * 1000)
        }

    def process(self, image_path, job_id):
        # Step 1: testing huruf latin
        latin_result = self._run_model(image_path, "latin")

        if latin_result["confidence"] >= self.LATIN_THRESHOLD:
            return self._format(job_id, latin_result)

        # Step 2: testing huruf jepang n cina
        japan_result = self._run_model(image_path, "japan")
        ch_result = self._run_model(image_path, "ch")

        # Pilih hasil dengan confidence tertinggi
        candidates = [latin_result, japan_result, ch_result]
        best = max(candidates, key=lambda x: x["confidence"])

        return self._format(job_id, best)

    def extract(self, image_path, job_id):
        # Backward compatible method
        return self.process(image_path, job_id)

    def _format(self, job_id, result):
        return {
            "job_id": job_id,
            "status": "completed",
            "text": result["text"],
            "languages": [result["model"]],
            "model_used": result["model"],
            "confidence": result["confidence"],
            "processing_time_ms": result["processing_time_ms"]
        }