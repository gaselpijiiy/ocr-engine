import re
import nltk
from nltk.tokenize import sent_tokenize
from difflib import SequenceMatcher


# INIT NLTK
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')


# BASIC CLEANING
def clean_text(text):
    # Gabung angka yang terpisah newline
    text = re.sub(r'(\d)\s*\n+\s*(\d)', r'\1\2', text)
    # Gabung kata huruf kecil yang terpisah newline (bukan kalimat baru)
    text = re.sub(r'([a-z])\n+([a-z])', r'\1 \2', text)

    # Tambah spasi setelah titik/koma yang menempel kata
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r',(\w)', r', \1', text)

    # Pisah angka dan huruf yang menyatu (2 pass)
    for _ in range(2):
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# FIX WORD MERGE
def fix_word_merge(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # atribut spacing (general)
    text = re.sub(r'(atribut)(\d+)', r'\1 \2', text)
    text = re.sub(r'([A-Z])atribut', r'\1 atribut', text)

    return text


# OCR FIXES
def fix_ocr_errors(text):
    common_fixes = {
        "ano": "and",
        "yearsold": "years old",
        "milesaway": "miles away",
        "yangdi": "yang di",
    }

    for wrong, correct in common_fixes.items():
        text = text.replace(wrong, correct)

    return text


# SINGKATAN
def normalize_abbreviations(text):
    abbreviations = {
        "No.": "No",
        "Dr.": "Dr",
        "Mr.": "Mr",
        "Mrs.": "Mrs",
        "Prof.": "Prof",
        "dll.": "dll",
        "dsb.": "dsb",
        "dst.": "dst",
    }

    for abbr, repl in abbreviations.items():
        text = text.replace(abbr, repl)

    return text


# FIX SENTENCE BOUNDARY
def fix_sentence_boundaries(text):
    """
    Menambahkan titik sebelum kata-kata yang secara kontekstual sebelum
    memulai kalimat baru (kata sambung pembuka paragraf dalam Bahasa Indonesia)
    """
    sentence_starters = [
        r'Pengujian', r'Hasil', r'Pada', r'Jika', r'Berdasarkan',
        r'Output', r'Dengan', r'Dikarenakan', r'Keluaran', r'Terdapat',
        r'Data', r'Kemampuan', r'Seluruh',
    ]

    for word in sentence_starters:
        # Tambah titik hanya jika didahului huruf kecil atau angka (bukan titik)
        text = re.sub(rf'([a-z0-9])\s+({word}\b)', r'\1. \2', text)

    return text



def fix_missing_periods_v2(text):
    return text  


# NORMALIZE PUNCTUATION
def normalize_punctuation(text):
    text = re.sub(r"[!]+", ".", text)       # ! → .
    text = re.sub(r"\s+\.", ".", text)      # spasi sebelum titik
    text = re.sub(r"\s+,", ",", text)       # spasi sebelum koma

    # rapikan 'None'
    text = re.sub(r"'\s*None\s*[!']*", "'None'", text)

    return text


def normalize_spacing(text):
    # pastikan titik diikuti spasi
    text = re.sub(r'\.(?!\s)', '. ', text)
    return text


# FIX LONG WORD
def fix_long_word_merge(text):
    words = text.split()
    fixed_words = []

    for w in words:
        if len(w) > 20:  # indikasi OCR merge
            split_w = re.sub(r'([a-z])([A-Z])', r'\1 \2', w)
            fixed_words.append(split_w)
        else:
            fixed_words.append(w)

    return " ".join(fixed_words)


# SENTENCE SPLIT
def split_sentences(text):
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 5]


# SIMILARITY-BASED MERGE
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def fuse_sentences(s1, s2):
    words1 = s1.split()
    words2 = s2.split()

    merged = []
    i, j = 0, 0

    while i < len(words1) and j < len(words2):
        if words1[i] == words2[j]:
            merged.append(words1[i])
            i += 1
            j += 1
        else:
            if len(words1[i]) > len(words2[j]):
                merged.append(words1[i])
                i += 1
            else:
                merged.append(words2[j])
                j += 1

    merged.extend(words1[i:])
    merged.extend(words2[j:])

    return " ".join(merged)


def filter_bad_sentences(sentences):
    result = []

    for s in sentences:
        if len(s.split()) < 4:
            continue
        if len(re.findall(r"[^\w\s\.,]", s)) > 10:
            continue
        result.append(s)

    return result


def select_best_sentences(sentences, threshold=0.65):
    clusters = []

    for sent in sentences:
        placed = False

        for cluster in clusters:
            if similarity(sent, cluster[0]) > threshold:
                cluster.append(sent)
                placed = True
                break

        if not placed:
            clusters.append([sent])

    result = []

    for cluster in clusters:
        def sentence_score(s):
            words = s.split()
            length_score = len(s)
            unique_score = len(set(words))
            noise_penalty = len(re.findall(r"[^\w\s\.,]", s))
            return length_score + unique_score * 2 - noise_penalty * 5

        best = max(cluster, key=sentence_score)
        result.append(best)

    return result


# GROUP PARAGRAPH
def group_into_paragraphs(sentences, max_sentences=3):
    paragraphs = []
    temp = []

    for sent in sentences:
        temp.append(sent)

        if len(temp) == max_sentences:
            paragraphs.append(" ".join(temp))
            temp = []

    if temp:
        paragraphs.append(" ".join(temp))

    return "\n\n".join(paragraphs)


# FINAL CLEANUP
def final_cleanup(text):
    # Bersihkan spasi berlebih di sekitar tanda baca
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])\s{2,}', r'\1 ', text)

    # Perbaiki titik yang hilang sebelum "Jika" yang didahului huruf kecil
    text = re.sub(r'([a-z])(Jika\b)', r'\1. \2', text)

    # Normalisasi spasi ganda
    text = re.sub(r' {2,}', ' ', text)

    return text


def deduplicate_sentences(sentences, threshold=0.85):
    result = []

    def sent_quality(s):
        words = s.split()
        score = len(s) + len(set(words)) * 2
        score -= len(re.findall(r"[^\w\s\.,]", s)) * 5
        return score

    for sent in sentences:
        duplicate_idx = None

        for i, existing in enumerate(result):
            sim = similarity(sent, existing)
            if sim > threshold:
                duplicate_idx = i
                break

            # Cek substring: kalimat pendek yang merupakan bagian dari kalimat panjang
            sent_norm = sent.lower().strip()
            exist_norm = existing.lower().strip()
            if len(sent_norm) > 20 and len(exist_norm) > 20:
                if abs(len(sent_norm) - len(exist_norm)) >= 30:
                    if sent_norm in exist_norm or exist_norm in sent_norm:
                        duplicate_idx = i
                        break

        if duplicate_idx is None:
            result.append(sent)
        else:
            # Pertahankan versi berkualitas lebih tinggi, di posisi aslinya
            if sent_quality(sent) > sent_quality(result[duplicate_idx]):
                result[duplicate_idx] = sent

    return result


# MAIN PIPELINE
def process_text(text):
    init_nltk()

    # 1. cleaning dasar
    text = clean_text(text)

    # 2. word-level correction
    text = fix_word_merge(text)
    text = fix_long_word_merge(text)
    text = fix_ocr_errors(text)

    # 3. normalization
    text = normalize_abbreviations(text)
    text = normalize_punctuation(text)

    # 4. sentence reconstruction
    text = fix_sentence_boundaries(text)
    text = normalize_spacing(text)

    # 5. split
    sentences = split_sentences(text)
    sentences = filter_bad_sentences(sentences)
    sentences = deduplicate_sentences(sentences)

    # 6. paragraf
    final_text = group_into_paragraphs(sentences, max_sentences=3)

    # 7. final cleanup
    final_text = final_cleanup(final_text)

    return final_text