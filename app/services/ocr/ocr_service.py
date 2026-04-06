import time
import cv2
import re
import numpy as np
import tempfile
import os
from paddleocr import PaddleOCR


class OCRService:
    PDF_DPI = 200  # akurat dan lebih cepat, DPI 300 ideal tapi lebih lambat
    POPPLER_PATH = r"D:\ocr-engine\poppler-25.12.0\Library\bin"

    def __init__(self):
        self.model_primary = PaddleOCR(
            use_angle_cls=True,
            lang="latin",
            det_db_thresh=0.1,
            det_db_box_thresh=0.1,
            det_db_unclip_ratio=1.8,
            det_limit_side_len=1920,
            det_limit_type='max',
            drop_score=0.3,
            rec_batch_num=6,
            max_text_length=200,
            use_gpu=False
        )

        # Fallback multi-language
        self.model_japan = PaddleOCR(use_angle_cls=True, lang="japan")
        self.model_ch    = PaddleOCR(use_angle_cls=True, lang="ch")

        self.LATIN_THRESHOLD = 0.80

    
    # PDF → IMAGE CONVERSION
    def _pdf_to_images(self, pdf_path):
        try:
            from pdf2image import convert_from_path
            kwargs = {"dpi": self.PDF_DPI}
            if self.POPPLER_PATH:
                kwargs["poppler_path"] = self.POPPLER_PATH
            return convert_from_path(pdf_path, **kwargs)
        except Exception as e:
            print(f"[OCR] PDF conversion failed: {e}")
            return None

    def _is_pdf(self, path):
        return path.lower().endswith(".pdf")


    # PREPROCESSING
    def _preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, image_path

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp  = cv2.filter2D(denoised, -1, kernel)

        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR), None

    def _preprocess_pil(self, pil_image):
        img_np = np.array(pil_image.convert("RGB"))
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    
    # SORTING BARIS
    def _sort_lines(self, result):
        if not result or not result[0]:
            return []

        lines = result[0]

        processed = []
        for line in lines:
            box = line[0]
            xs  = [p[0] for p in box]
            ys  = [p[1] for p in box]
            processed.append({
                "line":    line,
                "x_left":  min(xs),
                "x_right": max(xs),
                "y_top":   min(ys),
                "y_bot":   max(ys),
                "width":   max(xs) - min(xs),
            })

        if not processed:
            return []

        # Deteksi layout
        page_width   = max(b["x_right"] for b in processed)
        widths       = sorted(b["width"] for b in processed)
        median_width = widths[len(widths) // 2]
        is_single_column = (page_width > 0) and (median_width / page_width >= 0.55)

        # Sort by y_top dulu (berlaku untuk single maupun multi-column)
        processed.sort(key=lambda b: b["y_top"])

        if len(processed) < 2:
            return [b["line"] for b in processed]

        # Hitung threshold dari distribusi gap aktual
        y_tops = [b["y_top"] for b in processed]
        gaps = [y_tops[i+1] - y_tops[i] for i in range(len(y_tops)-1)]
        nonzero_gaps = [g for g in gaps if g > 0]

        if not nonzero_gaps:
            processed.sort(key=lambda b: b["x_left"])
            return [b["line"] for b in processed]

        median_gap = sorted(nonzero_gaps)[len(nonzero_gaps) // 2]

        # Threshold = 0.5x median_gap
        row_threshold = max(median_gap * 0.5, 4)

        # Build clusters sequentially berdasarkan gap
        row_clusters  = []
        current_cluster = [processed[0]]

        for box in processed[1:]:
            gap = box["y_top"] - current_cluster[-1]["y_top"]
            if gap <= row_threshold:
                current_cluster.append(box)
            else:
                row_clusters.append(current_cluster)
                current_cluster = [box]
        row_clusters.append(current_cluster)

        # Sort antar cluster top-to-bottom; dalam cluster left-to-right
        row_clusters.sort(key=lambda c: min(b["y_top"] for b in c))
        final_lines = []
        for cluster in row_clusters:
            cluster.sort(key=lambda b: b["x_left"])
            final_lines.extend(b["line"] for b in cluster)

        return final_lines

    
    # EKSTRAKSI TEKS
    def _extract_text(self, result):

        # DEBUG raw boxes sebelum sort
        if result and result[0]:
            print(f"[DEBUG] Raw boxes dari PaddleOCR: {len(result[0])}")
            for i, line in enumerate(result[0]):
                box   = line[0]
                y_top = min(p[1] for p in box)
                y_bot = max(p[1] for p in box)
                x_left = min(p[0] for p in box)
                text  = line[1][0]
                conf  = line[1][1]
                print(f"  raw[{i:02d}] y={y_top:.0f}-{y_bot:.0f} x={x_left:.0f} conf={conf:.2f} | {text[:50]}")
        

        lines = self._sort_lines(result)
        if not lines:
            return "", 0.0

        # DEBUG setelah sort
        print(f"[DEBUG] Total baris terdeteksi: {len(lines)}")
        for i, line in enumerate(lines):
            print(f"  [{i:02d}] conf={line[1][1]:.2f} | {line[1][0][:60]}")
        

        def is_hallucination(text):
            if len(text) < 5:
                return False
            # Satu karakter mendominasi > 40%
            for ch in set(text.lower()):
                if text.lower().count(ch) / len(text) > 0.4:
                    return True
            # Rasio karakter unik < 15%
            if len(set(text.lower())) / len(text) < 0.15:
                return True
            return False

        texts = []
        confs = []
        for line in lines:
            text = line[1][0]
            conf = line[1][1]
            if is_hallucination(text):
                print(f"[SKIP] hallucination terdeteksi (conf={conf:.2f}): '{text[:60]}'")
                continue
            texts.append(text)
            confs.append(conf)

        if not texts:
            return "", 0.0

        avg_conf = sum(confs) / len(confs)
        return " ".join(texts), avg_conf


    # POST-PROCESSING TEKS                                                
    def _post_process(self, text):
        text = self._fix_spacing(text)
        text = self._normalize_punctuation(text)
        text = self._deduplicate_sentences(text)
        return text.strip()

    def _fix_spacing(self, text):
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
        text = re.sub(r' {2,}', ' ', text)
        return text

    def _normalize_punctuation(self, text):
        text = re.sub(r'([.,!?])([^\s\d\'"])', r'\1 \2', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r"[''`]", "'", text)
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'\.{2,}', '.', text)
        return text

    def _deduplicate_sentences(self, text):
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

        if not sentences:
            return text

        def normalize(s):
            s = s.lower()
            s = re.sub(r'[^\w\s]', '', s)
            return re.sub(r'\s+', ' ', s).strip()

        def token_sim(a, b):
            ta = set(normalize(a).split())
            tb = set(normalize(b).split())
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta | tb)

        def is_substring(a, b):
            na, nb = normalize(a), normalize(b)
            if abs(len(na) - len(nb)) < 30:
                return False
            return na in nb or nb in na

        def quality_score(s):
            score = len(s) + s.count(' ') * 2
            score -= len(re.findall(r'[a-z][A-Z]', s)) * 5
            score -= len(re.findall(r'\d[a-zA-Z]|[a-zA-Z]\d', s))
            return score

        result = []
        for sent in sentences:
            dup_idx = None
            for i, existing in enumerate(result):
                if token_sim(sent, existing) >= 0.80 or is_substring(sent, existing):
                    dup_idx = i
                    break

            if dup_idx is None:
                result.append(sent)
            elif quality_score(sent) > quality_score(result[dup_idx]):
                result[dup_idx] = sent  # ganti di posisi asli, tidak reorder

        return ' '.join(result)


    # SINGLE-PASS OCR
    def _run_ocr_on_array(self, model, img_bgr):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, img_bgr)
            return model.ocr(tmp_path, cls=True)
        finally:
            os.unlink(tmp_path)

    def _run_single(self, model, image_path, model_name):
        start = time.time()

        if self._is_pdf(image_path):
            pages = self._pdf_to_images(image_path)
            if pages is None:
                # Fallback: biarkan PaddleOCR coba baca PDF langsung
                result = model.ocr(image_path, cls=True)
                text, conf = self._extract_text(result)
            else:
                all_texts = []
                all_confs = []
                for pil_img in pages:
                    img_bgr = self._preprocess_pil(pil_img)
                    result  = self._run_ocr_on_array(model, img_bgr)
                    t, c    = self._extract_text(result)
                    if t.strip():
                        all_texts.append(t)
                        all_confs.append(c)
                text = " ".join(all_texts)
                conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        else:
            img_bgr, fallback_path = self._preprocess_image(image_path)
            if img_bgr is not None:
                result = self._run_ocr_on_array(model, img_bgr)
            else:
                result = model.ocr(fallback_path, cls=True)
            text, conf = self._extract_text(result)

        return {
            "text":               text,
            "confidence":         round(conf, 3),
            "model":              model_name,
            "processing_time_ms": int((time.time() - start) * 1000),
        }


    # MAIN PROCESS
    def process(self, image_path, job_id):
        result = self._run_single(self.model_primary, image_path, "latin")

        if result["confidence"] < self.LATIN_THRESHOLD:
            japan  = self._run_single(self.model_japan, image_path, "japan")
            ch     = self._run_single(self.model_ch,    image_path, "ch")
            result = max([result, japan, ch], key=lambda x: x["confidence"])

        result["text"] = self._post_process(result["text"])
        return self._format(job_id, result)

    def extract(self, image_path, job_id):
        return self.process(image_path, job_id)


    # FORMAT OUTPUT
    def _format(self, job_id, result):
        return {
            "job_id":             job_id,
            "status":             "completed",
            "text":               result["text"],
            "languages":          [result["model"]],
            "model_used":         result["model"],
            "confidence":         result["confidence"],
            "processing_time_ms": result["processing_time_ms"],
        }