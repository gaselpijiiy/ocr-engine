import time
import cv2
import difflib
import re
from paddleocr import PaddleOCR


class OCRService:

    def __init__(self):
        """
        Smart Routing OCRService untuk global multi-language
        PaddleOCR, confidence-based, backward compatible
        """
        # Primary models
        self.model_primary = PaddleOCR(
            use_angle_cls=True,
            lang="latin",
            det_db_thresh=0.2,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=1.8,
            rec_batch_num=6,
            max_text_length=200,
            use_gpu=False
            )

        # loose model (high recall)
        self.model_loose = PaddleOCR(
            use_angle_cls=True,
            lang="latin",
            det_db_thresh=0.15,
            det_db_box_thresh=0.2,
            det_db_unclip_ratio=2.2,
            rec_batch_num=6,
            max_text_length=200,
            use_gpu=False
            )

        # Optional multi-language
        self.model_japan = PaddleOCR(use_angle_cls=True, lang="japan")
        self.model_ch = PaddleOCR(use_angle_cls=True, lang="ch")

        self.LATIN_THRESHOLD = 0.80


        # PREPROCESSING UNTUK MENINGKATKAN KUALITAS OCR #
    def _preprocess_image(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            return image_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # sharpening
        kernel = [[0, -1, 0],
                  [-1, 5, -1],
                  [0, -1, 0]]
        sharp = cv2.filter2D(gray, -1, kernel)

        # threshold
        _, thresh = cv2.threshold(sharp, 150, 255, cv2.THRESH_BINARY)

        return thresh

        # sort lines by Y-axis
    def _sort_lines(self, result):
        if not result or not result[0]:
            return []

        lines = result[0]
        lines.sort(key=lambda x: x[0][0][1])  # sort by Y axis
        return lines

    
    # extract text + confidence
    def _extract_text(self, result):
        lines = self._sort_lines(result)

        texts = []
        confs = []

        for line in lines:
            texts.append(line[1][0])
            confs.append(line[1][1])

        avg_conf = sum(confs) / len(confs) if confs else 0

        return " ".join(texts), avg_conf


    # MERGE RESULTS
    def _merge_results(self, text1, conf1, text2, conf2):
        def split_sentences(text):
            return [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 15]

        s1 = split_sentences(text1)
        s2 = split_sentences(text2)

        final = []

        used_s2 = set()

        for sent1 in s1:
            best_match = None
            best_score = 0
            best_idx = -1

            for i, sent2 in enumerate(s2):
                score = difflib.SequenceMatcher(None, sent1, sent2).ratio()
                if score > best_score:
                    best_score = score
                    best_match = sent2
                    best_idx = i

            # kalau mirip → pilih yang lebih "rapi"
            if best_score > 0.75:
                # scoring sederhana: penalti untuk kalimat jelek
                def score_sentence(s):
                    penalty = 0
                    if "Plumber," in s: penalty += 2
                    if "atributl=" in s: penalty += 2
                    if "menampilkanbeberapa" in s: penalty += 1
                    return len(s) - penalty

                chosen = sent1 if score_sentence(sent1) >= score_sentence(best_match) else best_match
                final.append(chosen)

                used_s2.add(best_idx)
            else:
                final.append(sent1)

        # tambahkan kalimat baru dari s2
        for i, sent2 in enumerate(s2):
            if i not in used_s2:
                if not any(difflib.SequenceMatcher(None, sent2, f).ratio() > 0.7 for f in final):
                    final.append(sent2)

        # join
        final_text = ". ".join(final)

        # cleanup ringan
        final_text = re.sub(r'\s+\.', '.', final_text)
        final_text = re.sub(r'\.\s*\.', '.', final_text)

        return final_text, max(conf1, conf2)

    
    # RUN MULTI-PASS OCR
    def _run_latin_multi_pass(self, image_path):
        start = time.time()

        img = self._preprocess_image(image_path)

        # Pass 1
        result_primary = self.model_primary.ocr(img, cls=True)
        text1, conf1 = self._extract_text(result_primary)

        # Pass 2
        result_loose = self.model_loose.ocr(img, cls=True)
        text2, conf2 = self._extract_text(result_loose)

        # Merge
        final_text, final_conf = self._merge_results(text1, conf1, text2, conf2)

        return {
            "text": final_text,
            "confidence": round(final_conf, 3),
            "model": "latin_multi_pass",
            "processing_time_ms": int((time.time() - start) * 1000)
        }

  
    # MAIN PROCESS #
    def process(self, image_path, job_id):

        # Step 1: multi-pass latin
        latin_result = self._run_latin_multi_pass(image_path)

        if latin_result["confidence"] >= self.LATIN_THRESHOLD:
            return self._format(job_id, latin_result)

        # Step 2: fallback language
        japan_result = self._run_single(self.model_japan, image_path, "japan")
        ch_result = self._run_single(self.model_ch, image_path, "ch")

        candidates = [latin_result, japan_result, ch_result]
        best = max(candidates, key=lambda x: x["confidence"])

        return self._format(job_id, best)


    # Single Model (fallback)
    def _run_single(self, model, image_path, model_name):
        start = time.time()

        result = model.ocr(image_path, cls=True)
        text, conf = self._extract_text(result)

        return {
            "text": text,
            "confidence": round(conf, 3),
            "model": model_name,
            "processing_time_ms": int((time.time() - start) * 1000)
        }

    # Backward Compability
    def extract(self, image_path, job_id):
        return self.process(image_path, job_id)


    # FORMAT OUTPUT #
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