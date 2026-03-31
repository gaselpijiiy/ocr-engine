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


# BASIC CLEANING
def clean_text(text):
    text = re.sub(r'(\d)\s*\n+\s*(\d)', r'\1\2', text)
    text = re.sub(r'([a-z])\n+([a-z])', r'\1 \2', text)

    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r',(\w)', r', \1', text)

    for _ in range(2):
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# FIX WORD MERGE
def fix_word_merge(text):
    # camelCase / OCR merge
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
    # huruf kecil → huruf besar = kemungkinan kalimat baru
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
    return text


# FIX MISSING PERIODS
def fix_missing_periods_v2(text):
    keywords = [
        "Pengujian", "Hasil", "Data", "Jika", "Pada",
        "Berdasarkan", "Output", "Dengan"
    ]

    for word in keywords:
        text = re.sub(rf'(?<!\.)\s+({word})', r'. \1', text)

    return text


# NORMALIZE PUNCTUATION
def normalize_punctuation(text):
    text = re.sub(r"[!]+", ".", text)       # ! → .
    text = re.sub(r"\s+\.", ".", text)      # spasi sebelum titik
    text = re.sub(r"\s+,", ",", text)       # spasi sebelum koma

    # rapikan 'None'
    text = re.sub(r"'\s*None\s*[!']*", "'None'", text)

    return text


# FIX LONG WORD
def fix_long_word_merge(text):
    # pisahin kata panjang aneh (heuristic)
    words = text.split()
    fixed_words = []

    for w in words:
        if len(w) > 20:  # indikasi OCR merge
            # coba pecah berdasarkan huruf besar
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
            # ambil kata yang lebih informatif (panjang)
            if len(words1[i]) > len(words2[j]):
                merged.append(words1[i])
                i += 1
            else:
                merged.append(words2[j])
                j += 1

    # sisa kata
    merged.extend(words1[i:])
    merged.extend(words2[j:])

    return " ".join(merged)

def filter_bad_sentences(sentences):
    result = []

    for s in sentences:
        # buang kalimat terlalu pendek / noise
        if len(s.split()) < 4:
            continue

        # buang kalau terlalu banyak simbol
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

            # penalti kalau terlalu banyak simbol aneh
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
    text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
    text = re.sub(r'(\w)(Jika)', r'\1. Jika', text)
    text = re.sub(r'menampilkanbeberapa', 'menampilkan beberapa', text)
    return text

def remove_duplicate_paragraphs(text, threshold=0.8):
    paragraphs = re.split(r'\n{2,}|\.\s+(?=[A-Z])', text)

    unique_paragraphs = []

    for p in paragraphs:
        p = p.strip()
        if len(p) < 50:
            continue

        is_duplicate = False

        for up in unique_paragraphs:
            if similarity(p, up) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_paragraphs.append(p)

    return "\n\n".join(unique_paragraphs)

# MAIN PIPELINE
def process_text(text):
    init_nltk()

    # REMOVE DUPLICATE PARAGRAPH
    text = remove_duplicate_paragraphs(text)

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
    text = fix_missing_periods_v2(text)
    text = fix_sentence_boundaries(text)

    # 5. split
    sentences = split_sentences(text)
    sentences = filter_bad_sentences(sentences)
    sentences = select_best_sentences(sentences)

    # 6. paragraf
    final_text = group_into_paragraphs(sentences, max_sentences=3)

    # 7. final cleanup
    final_text = final_cleanup(final_text)

    return final_text