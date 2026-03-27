import re
import nltk
from nltk.tokenize import sent_tokenize


# INIT NLTK (safe download)
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


# STEP 1: BASIC CLEANING
def clean_text(text):
    # Fix newline aneh
    text = re.sub(r'(\d)\s*\n+\s*(\d)', r'\1\2', text)
    text = re.sub(r'([a-z])\n+([a-z])', r'\1 \2', text)

    # Spasi setelah tanda baca
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r',(\w)', r', \1', text)

    # Pisahkan angka & huruf
    for _ in range(2):
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)

    # Rapikan spasi
    text = re.sub(r'\s+', ' ', text)

    return text.strip()



# STEP 2: OCR FIXES
def fix_ocr_errors(text):
    # Kata nempel (camelCase / lowercase-uppercase)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Error umum OCR
    common_fixes = {
        "ano": "and",
        "yearsold": "years old",
        "milesaway": "miles away",
    }

    for wrong, correct in common_fixes.items():
        text = text.replace(wrong, correct)

    return text


# STEP 3: SINGKATAN (ID + EN)
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

    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)

    return text


# STEP 4: SENTENCE SPLITTING
def split_sentences(text):
    sentences = sent_tokenize(text)
    return "\n".join(sentences)


# MAIN PIPELINE
def process_text(text):
    init_nltk()

    text = clean_text(text)
    text = fix_ocr_errors(text)
    text = normalize_abbreviations(text)
    text = split_sentences(text)

    return text