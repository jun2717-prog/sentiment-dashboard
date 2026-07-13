"""
AI-powered Review Analytics Dashboard — Python + Gradio
------------------------------------------------
Features:
  1. Multiple input options: single text, pasted multiple reviews, or file
     upload (.csv, .txt, .xlsx, .docx, .pdf, .json, .png/.jpg/.jpeg via OCR)
  2. Language detection
  3. Time-based trend analysis (uses a "date" column if present, else review order)
  4. Visualization dashboard (distribution + trend charts)
  5. Batch processing of many reviews at once
  6. Downloadable reports (CSV + PDF)

Run locally:
    pip install -r requirements.txt
    python app.py

Note: image upload (OCR) requires the Tesseract OCR program to be installed
on the machine, in addition to the Python packages. See README.md.
"""

import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime

import gradio as gr
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytesseract
from PIL import Image
from docx import Document as DocxDocument
from fpdf import FPDF
from langdetect import detect, LangDetectException
from pypdf import PdfReader
from transformers import pipeline

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
# 5-class model (Very Negative..Very Positive). Used here for English-language
# reviews only — we collapse its 5 classes down to 3 (Positive/Neutral/Negative)
# below to keep the app's labels consistent.
classifier = pipeline("sentiment-analysis", model=MODEL_NAME, top_k=None)

FIVE_CLASS_ORDER = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
FIVE_TO_THREE = {
    "very negative": "Negative", "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive", "very positive": "Positive",
}

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
}


def detect_language(text: str):
    """Best-effort language detection; returns (code, display_name).
    code is None on failure (very short/ambiguous text) — callers should
    treat that as "assume English" rather than blocking analysis."""
    try:
        code = detect(text)
        return code, LANGUAGE_NAMES.get(code, code)
    except LangDetectException:
        return None, "Unknown"


def _resolve_five_class_label(raw_label: str) -> str:
    """Handles both friendly labels ('Positive') and generic ones ('LABEL_3')."""
    label = raw_label.strip()
    if label.upper().startswith("LABEL_"):
        idx = int(label.split("_")[-1])
        label = FIVE_CLASS_ORDER[idx]
    return label


STAR_WEIGHTS = {
    "very negative": 1, "negative": 2, "neutral": 3, "positive": 4, "very positive": 5,
}


def translate_batch_to_english(texts: list) -> list:
    """Translates a batch of non-English texts to English, using the same
    Gemini setup as Ask AI, in as few API calls as possible (chunked to
    stay within safe prompt/output sizes — one call per ~20 reviews
    rather than one call per review, which matters for staying within
    the free tier's rate limit). Returns None entirely if no API key is
    set or the package isn't installed — callers should fall back to
    using the original text. If a chunk's API call fails or returns
    malformed data, that chunk falls back to the original (untranslated)
    text rather than losing data or crashing the whole batch."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[Translation] No GEMINI_API_KEY or GOOGLE_API_KEY found — skipping translation entirely.")
        return None
    try:
        from google import genai
    except ImportError:
        print("[Translation] google-genai package not installed — skipping translation entirely.")
        return None

    client = genai.Client(api_key=api_key)
    chunk_size = 20
    results = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        chunk_num = start // chunk_size + 1
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk))
        prompt = (
            "Translate each of the following numbered customer reviews to English. "
            "Reply with ONLY a JSON array of strings, one translation per review, "
            "in the exact same order and count as the input — no explanation, no "
            "markdown formatting, just the raw JSON array.\n\n"
            f"{numbered}"
        )
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(chunk):
                results.extend(str(t) for t in parsed)
            else:
                print(f"[Translation] Chunk {chunk_num}: unexpected response shape "
                      f"(expected {len(chunk)} items, got {len(parsed) if isinstance(parsed, list) else type(parsed).__name__}) "
                      f"— keeping original text for this chunk.")
                results.extend(chunk)  # unexpected shape — keep originals for this chunk
        except Exception as e:
            print(f"[Translation] Chunk {chunk_num} failed: {type(e).__name__}: {e} "
                  f"— keeping original text for this chunk.")
            results.extend(chunk)  # API/parse failure — keep originals for this chunk

    return results


def classify_text(text: str):
    """Returns (top_label, top_score, scores_dict, compound_score, star_rating).

    top_label/scores/compound: the model's 5 classes collapsed into 3
    (Positive/Neutral/Negative) by summing scores, so a 'Negative' +
    'Very Negative' combination can outweigh a single 'Positive' score
    even if no individual label does.

    star_rating: a 1.0-5.0 rating computed as a weighted average across
    all 5 raw classes (Very Negative=1 ... Very Positive=5), so it reflects
    the model's full confidence spread rather than just picking one bucket
    (e.g. 55% Positive + 45% Very Positive lands around 4.45, not a flat 4)."""
    raw = classifier(text[:2000])[0]  # cap length for speed/safety

    five_scores = {}
    for r in raw:
        five_class = _resolve_five_class_label(r["label"]).lower()
        five_scores[five_class] = five_scores.get(five_class, 0.0) + r["score"]

    star_rating = sum(five_scores.get(k, 0.0) * w for k, w in STAR_WEIGHTS.items())

    scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    for five_class, val in five_scores.items():
        bucket = FIVE_TO_THREE.get(five_class, "Neutral")
        scores[bucket] += val

    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    # Compound score: -1 (fully negative) to +1 (fully positive)
    compound = scores.get("Positive", 0) - scores.get("Negative", 0)
    return top_label, top_score, scores, compound, star_rating


# ---------------------------------------------------------------------------
# Aspect-based sentiment (price / quality / customer service / delivery /
# functionality / usability)
#
# NOTE ON APPROACH: this uses keyword matching to find which sentences
# mention each aspect, then runs the same sentiment model on just those
# sentences. It is NOT a fine-tuned aspect-based model (those aren't
# practically available pretrained for arbitrary custom categories across
# arbitrary custom categories) — it's a transparent heuristic. It will catch
# direct mentions ("customer service was great") but miss indirect ones
# ("nobody got back to me" without the phrase "customer service").
# ---------------------------------------------------------------------------

ASPECT_KEYWORDS = {
    "Price": {
        "en": ["price", "cost", "expensive", "cheap", "affordable", "value for money",
               "overpriced", "pricey", "worth the money", "price tag", "cost-effective"],
    },
    "Quality": {
        "en": ["quality", "well-made", "durable", "sturdy", "flimsy", "cheaply made",
               "build quality", "material", "craftsmanship"],
    },
    "Customer Service": {
        "en": ["customer service", "support team", "customer support", "responded",
               "response from", "help desk", "representative", "support staff"],
    },
    "Delivery": {
        "en": ["delivery", "shipping", "shipped", "arrived", "package arrived",
               "delayed", "on time", "courier", "delivery time"],
    },
    "Functionality": {
        "en": ["functionality", "feature", "functions", "performs", "malfunction",
               "does what it", "broken", "stopped working", "works as"],
    },
    "Usability": {
        "en": ["easy to use", "user-friendly", "intuitive", "difficult to use",
               "confusing", "navigate", "interface", "setup", "hard to use"],
    },
}
def star_to_bucket(v: float) -> str:
    """Maps a 1-5 star value back to a Positive/Neutral/Negative bucket,
    used when 'stars' is derived from aspect averages rather than the
    model's own 3-bucket classification, so the two stay consistent."""
    n = round(v)
    if n <= 2:
        return "Negative"
    if n == 3:
        return "Neutral"
    return "Positive"


def plurality_sentiment_label(sentiment_series: pd.Series) -> str:
    """Given a Series of Positive/Neutral/Negative labels, returns "Mostly
    X" for whichever is most common, or "Mixed" if there's no clear
    leader. Deliberately NOT based on averaging star values — averaging a
    group that's mostly Positive and Negative (few Neutral) would
    misleadingly land near the middle and say "Neutral" even when almost
    none of the items are actually neutral. Shared by both the overall
    summary and each aspect's summary, so they can't disagree."""
    total = len(sentiment_series)
    if total == 0:
        return "No data"
    counts = sentiment_series.value_counts()
    pcts = {
        lbl: counts.get(lbl, 0) / total * 100
        for lbl in ["Positive", "Neutral", "Negative"]
    }
    ranked = sorted(pcts.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_pct = ranked[0]
    second_pct = ranked[1][1]
    return f"Mostly {top_label}" if (top_pct - second_pct) >= 10 else "Mixed"


def overall_sentiment_label(df: pd.DataFrame) -> str:
    """"Overall sentiment" for the whole batch — see plurality_sentiment_label."""
    return plurality_sentiment_label(df["sentiment"]) if len(df) else "No data"


ASPECTS = list(ASPECT_KEYWORDS.keys())

# ---------------------------------------------------------------------------
# Top keyword extraction (Positive / Negative)
#
# This app assumes English-language review text throughout — keyword
# extraction, aspect detection, and stopword filtering are all English-only.
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset("""
a an the this that these those is are was were be been being am
i you he she it we they me him her us them my your his its our their
mine yours hers ours theirs and or but if so because as of to in on at
for with without from by about into over under again further then once
here there when where why how all any both each few more most other some
such no nor not only own same than too very s t can will just don should now
do does did doing have has had having
work works working worked use uses used using get gets getting got gotten
make makes made making go goes going went gone thing things item items
really also would could stuff lot lots much many product products
one two three four five six seven eight nine ten
seem seems feel feels felt fine
never always often rarely usually sometimes still already yet
""".split())


def _normalize_word(word: str) -> str:
    """Conservative suffix-stripping used only to GROUP word variants
    together for counting (e.g. 'crash'/'crashed'/'crashing'/'crashes' all
    map to the same group) — NOT used directly as the display label. See
    extract_top_keywords() for how the actual displayed word is chosen."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]  # doubled consonant, e.g. "shipping" -> "shipp" -> "ship"
        return stem
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]  # e.g. "stopped" -> "stopp" -> "stop"
        return stem
    if len(word) > 4 and word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh", "o")):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _english_only_text(df: pd.DataFrame, sentiment_label: str) -> pd.Series:
    """Returns review text filtered to only rows that are genuinely
    English — either originally English (no translation needed) or where
    translation actually succeeded (text differs from original_text).
    Rows where translation was attempted but failed (the fallback kept
    the untranslated original, so text == original_text) are excluded,
    since keyword extraction is English-only and would otherwise surface
    foreign function words (e.g. German "die", Spanish "que") as if they
    were meaningful English keywords."""
    subset = df[df["sentiment"] == sentiment_label]
    if "original_text" not in subset.columns:
        return subset["text"]
    is_english = subset["original_text"].isna() | (subset["text"] != subset["original_text"])
    return subset.loc[is_english, "text"]


def extract_top_keywords(texts, top_n: int = 12):
    """Returns [(word_or_phrase, count), ...] for the most frequent
    meaningful words and two-word phrases across the given texts.
    English-language reviews only — see module note above.

    How word variants are displayed: variants are grouped internally via
    _normalize_word (e.g. "shipped"/"shipping" group together), but the
    label actually shown is whichever ORIGINAL surface form was most
    common within that group — not an artificial stem. So a group that's
    mostly "shipping" displays as "shipping", while one that's mostly
    "crashed"/"crashes" displays as "crash". With very small samples where
    forms are exactly tied, the tie-break isn't meaningful; it becomes
    more reliable as more reviews are analyzed.

    How phrases work: frequent adjacent word pairs (e.g. "customer
    service") are surfaced as their own keyword. If a pair's two words
    almost always appear together (≥60% of each word's occurrences), the
    standalone words are dropped from the results so you see "customer
    service" once instead of "customer", "service", and "customer
    service" all separately."""
    stem_counts = Counter()
    stem_surface_forms = defaultdict(Counter)
    bigram_counts = Counter()

    for t in texts:
        raw_words = re.findall(r"[a-zA-ZÀ-ÿ']+", str(t).lower())
        kept_stems = []  # same length as raw_words; None marks a filtered-out slot
        for w in raw_words:
            if len(w) < 3 or w in _STOPWORDS:
                kept_stems.append(None)
                continue
            stem = _normalize_word(w)
            if stem in _STOPWORDS or len(stem) < 3:
                kept_stems.append(None)
                continue
            kept_stems.append(stem)
            stem_counts[stem] += 1
            stem_surface_forms[stem][w] += 1

        for i in range(len(kept_stems) - 1):
            if kept_stems[i] is not None and kept_stems[i + 1] is not None:
                bigram_counts[(kept_stems[i], kept_stems[i + 1])] += 1

    strong_bigrams = {pair: c for pair, c in bigram_counts.items() if c >= 2}

    absorbed = set()
    for (s1, s2), c in strong_bigrams.items():
        if stem_counts.get(s1, 0) and c / stem_counts[s1] >= 0.6:
            absorbed.add(s1)
        if stem_counts.get(s2, 0) and c / stem_counts[s2] >= 0.6:
            absorbed.add(s2)

    combined = Counter()
    for stem, c in stem_counts.items():
        if stem not in absorbed:
            combined[stem] = c
    for pair, c in strong_bigrams.items():
        combined[pair] = c

    def display_label(stem: str) -> str:
        forms = stem_surface_forms[stem]
        return forms.most_common(1)[0][0] if forms else stem

    results = []
    for key, count in combined.most_common(top_n):
        if isinstance(key, tuple):
            s1, s2 = key
            results.append((f"{display_label(s1)} {display_label(s2)}", count))
        else:
            results.append((display_label(key), count))

    return results


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str):
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def analyze_aspects(text: str) -> dict:
    """Returns {aspect: {'label': str, 'compound': float or None, 'stars': float or None, 'confidence': float or None}}.
    'Not mentioned' / None means no keyword for that aspect was found."""
    sentences = split_sentences(text) or [text]
    results = {}
    for aspect, lang_keywords in ASPECT_KEYWORDS.items():
        all_keywords = [kw.lower() for kws in lang_keywords.values() for kw in kws]
        matched = [s for s in sentences if any(kw in s.lower() for kw in all_keywords)]
        if not matched:
            results[aspect] = {"label": "Not mentioned", "compound": None, "stars": None, "confidence": None}
            continue
        label, top_score, _, compound, stars = classify_text(" ".join(matched))
        results[aspect] = {"label": label, "compound": compound, "stars": stars, "confidence": top_score}
    return results


# ---------------------------------------------------------------------------
# Input parsing (single text / pasted list / file upload)
# ---------------------------------------------------------------------------

TEXT_COLUMN_KEYWORDS = (
    "text", "review", "reviews", "comment", "comments", "feedback", "message",
    "content", "description", "body", "opinion", "notes", "remarks",
    "review_text", "comment_text", "customer_feedback",
)


def _guess_text_column(df: pd.DataFrame):
    """Finds the column most likely to hold free-form review text — first by
    name, then (if no name matches) by picking whichever text-like column
    has the longest average entry, since that's a much safer signal than
    just grabbing the first column blindly. Returns None if nothing
    plausible is found (e.g. a file of only numbers/IDs)."""
    exact = next(
        (c for c in df.columns if str(c).strip().lower() in TEXT_COLUMN_KEYWORDS), None
    )
    if exact:
        return exact

    candidates = []
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            avg_len = df[c].dropna().astype(str).str.len().mean()
            if pd.notna(avg_len) and avg_len >= 15:  # short strings are more likely IDs/categories, not review text
                candidates.append((c, avg_len))

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[1])[0]


DATE_COLUMN_KEYWORDS = (
    "date", "timestamp", "time", "datetime", "created_at", "created",
    "purchase_date", "order_date", "review_date", "submitted", "submitted_at",
    "posted", "posted_at", "review_time",
)


def _guess_date_column(df: pd.DataFrame, exclude: str):
    """Finds a date column by name first, then (if nothing matches) by
    actually trying to parse each remaining column as a date and picking
    whichever one mostly succeeds — tolerant of unconventional names like
    'order_date' or 'submitted_on' that aren't in the keyword list."""
    exact = next(
        (c for c in df.columns if str(c).strip().lower() in DATE_COLUMN_KEYWORDS), None
    )
    if exact:
        return exact

    best_col, best_ratio = None, 0.0
    for c in df.columns:
        if c == exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue  # avoids misreading small ints (e.g. a 1-5 rating) as epoch timestamps
        parsed = pd.to_datetime(df[c], errors="coerce")
        non_null = df[c].notna().sum()
        if non_null == 0:
            continue
        success_ratio = parsed.notna().sum() / non_null
        if success_ratio > 0.8 and success_ratio > best_ratio:
            best_col, best_ratio = c, success_ratio
    return best_col


def _extract_text_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Shared column-detection logic for any tabular source (csv/xlsx/json-records).
    Tolerant of files that don't have every expected column — only 'text'
    (in some recognizable or guessable form) is required; 'date' and
    anything else is optional and simply ignored if absent."""
    text_col = _guess_text_column(df)
    if text_col is None:
        raise ValueError(
            "No review column detected. Please upload a CSV or Excel file "
            "containing customer reviews — or rename the column with your "
            "review text to 'text' (or 'review', 'comment', 'feedback', etc.)."
        )
    df = df.rename(columns={text_col: "text"})
    date_col = _guess_date_column(df, exclude="text")
    if date_col:
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["text"] = df["text"].astype(str)
    keep = ["text"] + (["date"] if "date" in df.columns else [])
    return df[keep].dropna(subset=["text"])


def _lines_to_dataframe(lines) -> pd.DataFrame:
    lines = [line.strip() for line in lines if line and line.strip()]
    if not lines:
        raise ValueError("The uploaded file contains no readable reviews.")
    return pd.DataFrame({"text": lines})


def _parse_csv(path) -> pd.DataFrame:
    return _extract_text_date_columns(pd.read_csv(path))


def _parse_xlsx(path) -> pd.DataFrame:
    return _extract_text_date_columns(pd.read_excel(path))


def _parse_txt(path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _lines_to_dataframe(f.readlines())


def _parse_docx(path) -> pd.DataFrame:
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return _lines_to_dataframe(paragraphs)


def _parse_pdf(path) -> pd.DataFrame:
    reader = PdfReader(path)
    lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines.extend(text.splitlines())
    return _lines_to_dataframe(lines)


def _parse_image(path) -> pd.DataFrame:
    try:
        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang="eng")
    except pytesseract.TesseractNotFoundError:
        raise ValueError(
            "Image upload needs the Tesseract OCR program installed on this machine "
            "(it's separate from the Python packages). See the README for install steps."
        )
    return _lines_to_dataframe(text.splitlines())


def _parse_json(path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    # Case 1: a plain list of strings, e.g. ["great!", "terrible."]
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return _lines_to_dataframe(data)

    # Case 2: a list of records/objects, e.g. [{"text": "...", "date": "..."}, ...]
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return _extract_text_date_columns(pd.DataFrame(data))

    # Case 3: a dict wrapping a list under some key, e.g. {"reviews": [...]}
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value:
                if all(isinstance(item, str) for item in value):
                    return _lines_to_dataframe(value)
                if all(isinstance(item, dict) for item in value):
                    return _extract_text_date_columns(pd.DataFrame(value))

    raise ValueError(
        "Couldn't find review text in this JSON file. Use a list of strings, "
        "a list of {\"text\": ...} objects, or a dict containing one of those lists."
    )


FILE_PARSERS = {
    ".csv": _parse_csv,
    ".xlsx": _parse_xlsx,
    ".txt": _parse_txt,
    ".docx": _parse_docx,
    ".pdf": _parse_pdf,
    ".json": _parse_json,
    ".png": _parse_image,
    ".jpg": _parse_image,
    ".jpeg": _parse_image,
}


# ---------------------------------------------------------------------------
# Input parsing (single text / pasted list / file upload)
# ---------------------------------------------------------------------------

def build_dataframe(mode: str, single_text: str, multi_text: str, uploaded_file) -> pd.DataFrame:
    """Turns whichever input mode was used into a standard DataFrame with a
    'text' column and an optional 'date' column."""

    if mode == "Single text":
        if not single_text or not single_text.strip():
            raise ValueError("Please enter some text.")
        return pd.DataFrame({"text": [single_text.strip()]})

    if mode == "Multiple reviews (one per line)":
        if not multi_text or not multi_text.strip():
            raise ValueError("Please paste at least one line of text.")
        lines = [line.strip() for line in multi_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Please paste at least one line of text.")
        return pd.DataFrame({"text": lines})

    if mode == "Upload file":
        if uploaded_file is None:
            raise ValueError("Please upload a file (.csv, .txt, .xlsx, .docx, .pdf, .json, .png, .jpg, or .jpeg).")
        path = uploaded_file.name
        suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        parser = FILE_PARSERS.get(suffix)
        if parser is None:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Please upload .csv, .txt, .xlsx, "
                ".docx, .pdf, .json, .png, .jpg, or .jpeg."
            )
        try:
            return parser(path)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Couldn't read this {suffix} file: {e}")

    raise ValueError("Unknown input mode.")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run_batch_analysis(mode, single_text, multi_text, uploaded_file, progress=gr.Progress()):
    try:
        progress(0, desc="Reading input...")
        df = build_dataframe(mode, single_text, multi_text, uploaded_file)
    except ValueError as e:
        return (
            pd.DataFrame({"Error": [str(e)]}),
            "", None, None, None, None, None, None,
            "Fix the input above and try again.",
            None, None,
        )

    # Detect language for every review (fast, local, free). Reviews under
    # 15 characters are treated as English regardless of what's detected —
    # language detection on very short text is unreliable, and a false
    # positive here would waste an API call (and risk garbling text that
    # was already fine) translating something like "Great!" unnecessarily.
    progress(0.02, desc="Detecting languages...")
    lang_codes, lang_names = [], []
    for text in df["text"]:
        if len(str(text).strip()) < 15:
            lang_codes.append("en")
            lang_names.append("English")
        else:
            code, name = detect_language(text)
            lang_codes.append(code or "en")
            lang_names.append(name if code else "English")
    df["language"] = lang_names

    # Translate only the reviews that actually need it, in as few batched
    # API calls as possible. original_text stays blank for reviews that
    # were already English (nothing was translated, nothing to show).
    df["original_text"] = None
    non_english_positions = [i for i, c in enumerate(lang_codes) if c != "en"]
    if non_english_positions:
        progress(0.04, desc=f"Translating {len(non_english_positions)} non-English review(s)...")
        originals = [df["text"].iloc[i] for i in non_english_positions]
        translated = translate_batch_to_english(originals)
        if translated is not None:
            for pos, orig, new_text in zip(non_english_positions, originals, translated):
                label = df.index[pos]
                df.at[label, "original_text"] = orig
                df.at[label, "text"] = new_text
        # If translated is None (no GEMINI_API_KEY set), original non-English
        # text is left as-is — the sentiment model is inherently multilingual
        # under the hood and will still attempt a best-effort rating, but
        # aspect detection, keyword extraction, and the AI summary are tuned
        # for English and won't work reliably without translation.

    labels, stars_list, confidence_list = [], [], []
    aspect_stars = {a: [] for a in ASPECTS}

    total_reviews = len(df)
    for i, text in enumerate(df["text"]):
        progress(0.05 + 0.9 * (i + 1) / total_reviews, desc=f"Analyzing review {i + 1} of {total_reviews}...")
        whole_label, whole_score, _, _, whole_stars = classify_text(text)

        aspects = analyze_aspects(text)
        aspect_star_values = []  # only genuinely mentioned aspects feed the main rating
        aspect_confidence_values = []
        for a in ASPECTS:
            s = aspects[a]["stars"]
            if s is not None:
                aspect_star_values.append(s)
                aspect_confidence_values.append(aspects[a]["confidence"])
                aspect_stars[a].append(round(s, 1))
            else:
                # Not explicitly discussed: leave this aspect blank
                # ("Not mentioned") rather than guessing a value for it.
                aspect_stars[a].append(None)

        if aspect_star_values:
            # At least one aspect was mentioned: the overall rating is the
            # average of the aspects actually discussed, not a separate
            # whole-text classification. Confidence follows the same path —
            # averaged across just those aspects, not the whole review.
            final_stars = sum(aspect_star_values) / len(aspect_star_values)
            final_confidence = sum(aspect_confidence_values) / len(aspect_confidence_values)
        else:
            # No aspects mentioned: fall back to the whole-text star rating
            # and the whole-text classification confidence.
            final_stars = whole_stars
            final_confidence = whole_score

        # "sentiment" is always derived from the same star number shown
        # everywhere else (star histogram, average_stars column) — never
        # from a separate model classification — so the two can't disagree.
        final_label = star_to_bucket(final_stars)

        labels.append(final_label)
        stars_list.append(round(final_stars, 1))
        confidence_list.append(round(final_confidence, 3))

    df["stars"] = stars_list
    df["sentiment"] = labels
    df["confidence"] = confidence_list

    for a in ASPECTS:
        stars_col = a.lower().replace(" ", "_") + "_stars"
        df[stars_col] = aspect_stars[a]

    # Put "stars" first and "sentiment" right after it (stars = main display,
    # sentiment = secondary label), ahead of everything else.
    lead_cols = ["text", "stars", "sentiment"]
    other_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + other_cols]

    progress(1.0, desc="Building dashboard...")
    pie_fig = make_sentiment_pie_chart(df)
    count_fig = make_sentiment_count_chart(df)
    aspect_fig = make_aspect_chart(df)

    pos_keywords = extract_top_keywords(_english_only_text(df, "Positive"), top_n=12)
    neg_keywords = extract_top_keywords(_english_only_text(df, "Negative"), top_n=12)
    pos_keywords_fig = make_keyword_chart(pos_keywords, "Positive", "#3E7C59")
    neg_keywords_fig = make_keyword_chart(neg_keywords, "Negative", "#B23A2E")
    trend_line_fig = make_sentiment_trend_line_chart(df)
    stat_cards_html = make_stat_cards_html(df)

    total = len(df)
    dist_counts = df["sentiment"].value_counts()
    pos_n, neu_n, neg_n = dist_counts.get("Positive", 0), dist_counts.get("Neutral", 0), dist_counts.get("Negative", 0)
    pos_pct, neu_pct, neg_pct = (pos_n / total * 100, neu_n / total * 100, neg_n / total * 100) if total else (0, 0, 0)

    overall = overall_sentiment_label(df)

    summary_lines = [
        "### AI Summary",
        f"**Overall sentiment:** {overall} "
        f"({pos_pct:.0f}% positive, {neg_pct:.0f}% negative, {neu_pct:.0f}% neutral, {total} reviews)",
    ]

    if pos_keywords:
        summary_lines.append("\n**Strengths:**")
        for word, _ in pos_keywords[:5]:
            summary_lines.append(f"- {word.capitalize()}")
    if neg_keywords:
        summary_lines.append("\n**Issues:**")
        for word, _ in neg_keywords[:5]:
            summary_lines.append(f"- {word.capitalize()}")

    summary_text = "\n".join(summary_lines)

    def _safe_truncate(series, length=90):
        return series.apply(
            lambda x: (str(x)[:length] + "…") if pd.notna(x) and len(str(x)) > length else x
        )

    full_display_df = df.copy()
    full_display_df["text"] = _safe_truncate(full_display_df["text"])
    full_display_df["original_text"] = _safe_truncate(full_display_df["original_text"])
    full_display_df["confidence"] = (full_display_df["confidence"] * 100).round().astype(int).astype(str) + "%"

    # Aspect columns (price_stars, quality_stars, etc.) and the numeric
    # "stars" column are kept in the underlying data — the Dashboard's
    # charts and the PDF report still use them — but dropped from the
    # table and CSV export. "sentiment" (Positive/Neutral/Negative) and
    # "confidence" are the ratings shown here now.
    aspect_stars_cols = [a.lower().replace(" ", "_") + "_stars" for a in ASPECTS]
    full_display_df = full_display_df.drop(columns=aspect_stars_cols + ["stars"])

    # Put original_text right after text, so a translated review and its
    # original sit next to each other for easy comparison.
    lead_cols = ["text", "original_text", "sentiment", "confidence"]
    other_cols = [c for c in full_display_df.columns if c not in lead_cols]
    full_display_df = full_display_df[lead_cols + other_cols]

    return (
        full_display_df, stat_cards_html, pie_fig, count_fig, aspect_fig,
        pos_keywords_fig, neg_keywords_fig, trend_line_fig,
        summary_text, df, None,
    )


def make_stat_cards_html(df: pd.DataFrame) -> str:
    """Four summary cards: Total / Positive / Negative / Neutral counts."""
    total = len(df)
    counts = df["sentiment"].value_counts()
    pos, neu, neg = counts.get("Positive", 0), counts.get("Neutral", 0), counts.get("Negative", 0)

    def card(label, value, color):
        return f"""
        <div style="flex:1; min-width:120px; background:{color}15; border:1px solid {color}40;
                    border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:13px; color:#666; margin-bottom:6px;">{label}</div>
            <div style="font-size:28px; font-weight:700; color:{color};">{value}</div>
        </div>
        """

    return f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:8px;">
        {card("Total Reviews", total, "#1B2420")}
        {card("Positive", pos, "#3E7C59")}
        {card("Negative", neg, "#B23A2E")}
        {card("Neutral", neu, "#B8935A")}
    </div>
    """


def make_sentiment_pie_chart(df: pd.DataFrame):
    """Percentage breakdown of Positive/Neutral/Negative."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    colors = {"Positive": "#3E7C59", "Neutral": "#B8935A", "Negative": "#B23A2E"}
    ax.pie(
        counts,
        labels=[f"{k} ({int(v)})" for k, v in counts.items()],
        colors=[colors[k] for k in counts.index],
        autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        startangle=90,
    )
    ax.set_title("Sentiment breakdown (%)", fontsize=10)
    fig.tight_layout()
    return fig


def make_sentiment_count_chart(df: pd.DataFrame):
    """How many Positive/Neutral/Negative reviews, as a bar chart."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
    colors = {"Positive": "#3E7C59", "Neutral": "#B8935A", "Negative": "#B23A2E"}
    bars = ax.bar(counts.index, counts.values, color=[colors[k] for k in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.02, str(v), ha="center", fontsize=10)
    ax.set_ylabel("Number of reviews")
    ax.set_title("Sentiment breakdown (count)", fontsize=10)
    ax.set_ylim(0, max(counts.values) * 1.15 if counts.values.max() > 0 else 1)
    fig.tight_layout()
    return fig


def make_aspect_chart(df: pd.DataFrame):
    """Sentiment composition (Positive/Neutral/Negative) per aspect, as a
    stacked horizontal bar — no average star number, just how many mentions
    of each aspect were positive vs. negative vs. neutral."""
    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = {"Positive": "#3E7C59", "Neutral": "#B8935A", "Negative": "#B23A2E"}

    names, pos_counts, neu_counts, neg_counts = [], [], [], []
    for a in ASPECTS:
        col = a.lower().replace(" ", "_") + "_stars"
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        buckets = vals.apply(star_to_bucket)
        names.append(a)
        pos_counts.append((buckets == "Positive").sum())
        neu_counts.append((buckets == "Neutral").sum())
        neg_counts.append((buckets == "Negative").sum())

    if not names:
        ax.text(0.5, 0.5, "No aspects were mentioned in this batch", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    y = range(len(names))
    ax.barh(y, pos_counts, color=colors["Positive"], label="Positive")
    ax.barh(y, neu_counts, left=pos_counts, color=colors["Neutral"], label="Neutral")
    left_neg = [p + n for p, n in zip(pos_counts, neu_counts)]
    ax.barh(y, neg_counts, left=left_neg, color=colors["Negative"], label="Negative")

    # Label each segment with its own count, centered inside that segment
    # (skipping segments with 0, since there's nothing to label there).
    for i in range(len(names)):
        if pos_counts[i] > 0:
            ax.text(pos_counts[i] / 2, i, str(pos_counts[i]), ha="center", va="center",
                     color="white", fontsize=8, fontweight="bold")
        if neu_counts[i] > 0:
            ax.text(pos_counts[i] + neu_counts[i] / 2, i, str(neu_counts[i]), ha="center", va="center",
                     color="white", fontsize=8, fontweight="bold")
        if neg_counts[i] > 0:
            ax.text(left_neg[i] + neg_counts[i] / 2, i, str(neg_counts[i]), ha="center", va="center",
                     color="white", fontsize=8, fontweight="bold")

    totals = [p + n + g for p, n, g in zip(pos_counts, neu_counts, neg_counts)]
    for i, total in enumerate(totals):
        ax.text(total + max(totals) * 0.02, i, f"n={total}", va="center", fontsize=8)

    ax.set_xlim(0, max(totals) * 1.18)  # extra right margin so "n=" labels never hit the plot edge
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("Number of mentions")
    ax.set_title("Sentiment by aspect (mentions only)")
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    return fig


def make_keyword_chart(top_keywords, sentiment_label: str, color: str):
    """Renders a precomputed [(word, count), ...] list (see extract_top_keywords)
    as a horizontal bar chart. Kept separate from extraction so the same
    keyword list can also feed the structured summary without recomputing."""
    fig, ax = plt.subplots(figsize=(4.5, 5))

    if not top_keywords:
        ax.text(0.5, 0.5, "No keywords found", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    words, counts = zip(*reversed(top_keywords))  # reversed so the top word ends up at the top of the barh
    ax.barh(words, counts, color=color)
    for i, c in enumerate(counts):
        ax.text(c + max(counts) * 0.02, i, str(c), va="center", fontsize=8)
    ax.set_xlabel("Mentions")
    ax.set_title(f"Top keywords — {sentiment_label}", fontsize=10)
    fig.tight_layout()
    return fig


def make_sentiment_trend_line_chart(df: pd.DataFrame):
    """Interactive (Plotly) line chart: count of Positive/Neutral/Negative
    reviews per week, oldest to newest. Weekly (not daily) aggregation is
    the default — with typically-sparse review data, a daily chart is
    mostly noise (constant 0/1 zigzag) rather than a readable trend.
    Returns a Plotly figure with a "no date data" message baked in if
    there's no usable date column, since gr.Plot needs something to
    render either way."""
    if "date" not in df.columns or df["date"].dropna().empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No date information available to display sentiment trends.",
            showarrow=False, font=dict(size=14),
            xref="paper", yref="paper", x=0.5, y=0.5,
        )
        fig.update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            height=350, title="Weekly Sentiment Trends",
        )
        return fig

    plot_df = df.dropna(subset=["date"]).copy()
    # Group by calendar week (Monday-start), using each week's start date
    # as the x-axis position so points land in clean, evenly-spaced order.
    plot_df["week_start"] = plot_df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    grouped = plot_df.groupby(["week_start", "sentiment"]).size().unstack(fill_value=0)
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped = grouped.sort_index()  # chronological, oldest to newest

    fig = go.Figure()
    line_colors = {"Positive": "#3E7C59", "Neutral": "#808080", "Negative": "#B23A2E"}
    for label in ["Positive", "Neutral", "Negative"]:
        fig.add_trace(go.Scatter(
            x=grouped.index, y=grouped[label],
            mode="lines", name=label,
            line=dict(color=line_colors[label], width=3, shape="spline", smoothing=0.6),
            hovertemplate=f"{label}: " + "%{y}<extra></extra>",
        ))

    fig.update_layout(
        title="Weekly Sentiment Trends",
        xaxis_title="Date", yaxis_title="Number of reviews",
        hovermode="x unified", height=400,
        font=dict(size=13),
        xaxis=dict(tickfont=dict(size=13), hoverformat="%Y-%m-%d"),
        yaxis=dict(tickfont=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

PAGE_SIZE = 15


def paginate_table(full_display_df, search_text, sentiment_filter, sort_order, page):
    """Applies search + sentiment filter + date sort to the full simplified
    results table, then slices out one page. Returns (page_df, status_text,
    clamped_page_number). Pure function of the already-computed data — no
    re-running of the model, so this is cheap to call on every UI change."""
    if full_display_df is None or len(full_display_df) == 0:
        return pd.DataFrame(), "No results yet.", 1

    working = full_display_df.copy()

    if search_text and search_text.strip():
        # Matches reviews containing ALL typed words, in any order — so
        # "shipping slow" finds "the shipping was very slow" even though
        # that's not one literal phrase match.
        search_words = search_text.strip().lower().split()
        text_lower = working["text"].str.lower()
        mask = pd.Series(True, index=working.index)
        for word in search_words:
            mask &= text_lower.str.contains(word, na=False, regex=False)
        working = working[mask]

    if sentiment_filter and sentiment_filter != "All":
        working = working[working["sentiment"] == sentiment_filter]

    if "date" in working.columns and sort_order in ("Newest first", "Oldest first"):
        working = working.sort_values("date", ascending=(sort_order == "Oldest first"), na_position="last")

    total_rows = len(working)
    total_pages = max(1, -(-total_rows // PAGE_SIZE))  # ceiling division
    page = max(1, min(int(page or 1), total_pages))
    start = (page - 1) * PAGE_SIZE
    page_df = working.iloc[start:start + PAGE_SIZE]

    status = f"Page {page} of {total_pages} ({total_rows} result{'s' if total_rows != 1 else ''})"
    return page_df, status, page


def generate_csv_report(df: pd.DataFrame):
    """Exports exactly what the Analyze tab table shows — same columns,
    same drops (no raw 'stars' or per-aspect '_stars' columns), and
    confidence shown as a percentage to match the on-screen table."""
    if df is None or df.empty:
        return None
    aspect_stars_cols = [a.lower().replace(" ", "_") + "_stars" for a in ASPECTS]
    drop_cols = [c for c in aspect_stars_cols + ["stars"] if c in df.columns]
    export_df = df.drop(columns=drop_cols)
    if "confidence" in export_df.columns:
        export_df = export_df.copy()
        export_df["confidence"] = (export_df["confidence"] * 100).round().astype(int).astype(str) + "%"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="sentiment_report_")
    export_df.to_csv(tmp.name, index=False)
    return tmp.name


def _pdf_safe(text: str) -> str:
    """FPDF's built-in fonts only support Latin-1. English text is always
    safe, but this stays as cheap insurance against stray characters some
    text editors insert (smart quotes, em-dashes, etc.) so a rogue
    character can't crash PDF generation."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _embed_pdf_chart(pdf: FPDF, fig, target_height: float = 75):
    """Saves a matplotlib figure to a temp PNG and embeds it in the PDF at
    a consistent HEIGHT (not a fixed width) — charts have different native
    aspect ratios (e.g. the wide aspect-breakdown chart vs. the more
    square pie chart), so scaling them all to the same width made them
    wildly different heights. Computing width from each figure's own
    aspect ratio keeps every chart visually the same size. Also forces
    the cursor back to the left margin after — fpdf2 doesn't reliably
    reset it after an image, which can leave later multi_cell() calls
    thinking there's no width left (this crashed the report before)."""
    fig_w_in, fig_h_in = fig.get_size_inches()
    width = target_height * (fig_w_in / fig_h_in)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=150)
    img_buf.seek(0)
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_img.write(img_buf.read())
    tmp_img.close()
    pdf.image(tmp_img.name, w=width)
    pdf.set_x(pdf.l_margin)
    pdf.ln(4)


def generate_pdf_report(df: pd.DataFrame, figs: dict):
    """figs: dict of chart name -> matplotlib Figure (e.g. {'pie': ...,
    'count': ..., 'aspect': ..., 'pos_keywords': ..., 'neg_keywords': ...}).
    Missing/None entries are just skipped, so this works even if only some
    charts were generated."""
    if df is None or df.empty:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Sentiment Analysis Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Total items analyzed: {len(df)}", ln=True)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 9, f"Overall sentiment: {overall_sentiment_label(df)}", ln=True)
    pdf.ln(2)

    counts = df["sentiment"].value_counts()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for lbl in ["Positive", "Neutral", "Negative"]:
        count = counts.get(lbl, 0)
        pct = (count / len(df)) * 100
        pdf.cell(0, 7, f"{lbl}: {count} ({pct:.1f}%)", ln=True)
    pdf.ln(4)

    # Embed every chart that was actually generated, stacked vertically.
    # Skips gracefully if a chart wasn't computed (figs.get returns None).
    for key in ("pie", "count", "aspect", "pos_keywords", "neg_keywords"):
        fig = figs.get(key)
        if fig is not None:
            _embed_pdf_chart(pdf, fig)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sentiment by aspect", ln=True)
    pdf.set_font("Helvetica", "", 9)
    any_aspect_mentioned = False
    for a in ASPECTS:
        col = a.lower().replace(" ", "_") + "_stars"
        vals = df[col].dropna()
        if len(vals) == 0:
            pdf.cell(0, 6, f"{a}: not mentioned in this batch", ln=True)
            continue
        any_aspect_mentioned = True
        # Bucket each individual mention first (matching exactly how the
        # aspect chart counts them), THEN find the plurality — not an
        # average of star numbers, which is what produced the "every
        # aspect says Neutral" bug this replaced.
        buckets = vals.apply(star_to_bucket)
        label = plurality_sentiment_label(buckets)
        counts = buckets.value_counts()
        pos_pct = counts.get("Positive", 0) / len(vals) * 100
        neu_pct = counts.get("Neutral", 0) / len(vals) * 100
        neg_pct = counts.get("Negative", 0) / len(vals) * 100
        pdf.cell(
            0, 6,
            f"{a}: {label} ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative, "
            f"{neu_pct:.0f}% neutral; mentioned in {len(vals)} of {len(df)} items)",
            ln=True,
        )
    if not any_aspect_mentioned:
        pdf.cell(0, 6, "No aspect keywords were detected in this batch.", ln=True)
    pdf.ln(2)

    # De-duplicate identical review text (common with repeated/templated
    # data) so the same line doesn't show up twice in these lists.
    unique_df = df.drop_duplicates(subset="text")

    def _is_pdf_renderable(text) -> bool:
        """True if this text can actually display in the PDF (Latin-1).
        Translation should normally have already converted everything to
        English before this point, but if it silently failed for some
        reason (e.g. the Gemini quota was exhausted at analysis time, so
        translate_batch_to_english fell back to the original text), this
        stops the PDF from showing a garbled wall of '?' characters —
        it picks a different, genuinely readable example instead."""
        try:
            str(text).encode("latin-1")
            return True
        except UnicodeEncodeError:
            return False

    def _render_example_line(row):
        text = _pdf_safe(str(row["text"])[:150].replace("\n", " "))
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, f"- {text}")

    def _pick_examples(sentiment_label):
        candidates = unique_df[unique_df["sentiment"] == sentiment_label]
        renderable = candidates[candidates["text"].apply(_is_pdf_renderable)]
        # Prefer genuinely readable examples; only fall back to whatever's
        # left (which may render as "?"s) if literally nothing is readable.
        pool = renderable if len(renderable) > 0 else candidates
        return pool.head(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Example Positive Reviews", ln=True)
    pdf.set_font("Helvetica", "", 9)
    positive_examples = _pick_examples("Positive")
    if len(positive_examples) == 0:
        pdf.cell(0, 6, "No positive reviews in this batch.", ln=True)
    else:
        for _, row in positive_examples.iterrows():
            _render_example_line(row)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Example Negative Reviews", ln=True)
    pdf.set_font("Helvetica", "", 9)
    negative_examples = _pick_examples("Negative")
    if len(negative_examples) == 0:
        pdf.cell(0, 6, "No negative reviews in this batch.", ln=True)
    else:
        for _, row in negative_examples.iterrows():
            _render_example_line(row)

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="sentiment_report_")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name


# ---------------------------------------------------------------------------
# Ask AI About the Reviews
#
# Uses Google's Gemini API, which has a genuine free tier (rate-limited —
# roughly 15 requests/minute and 1,500/day for Flash models as of this
# writing, no credit card required) — unlike OpenAI's API, this can run at
# zero cost for typical demo/portfolio use, consistent with the rest of
# this app. Very heavy usage could eventually hit the free quota and need
# billing enabled, but normal use should stay free.
#
# RETRIEVAL APPROACH: this uses simple keyword matching (reusing the same
# stopword list and aspect keyword lists as the rest of the app) to find
# relevant reviews, not a vector database. That's a deliberate simplicity
# tradeoff — it's cheap, fast, and needs no extra infrastructure, but it
# can miss reviews that are relevant in meaning without sharing literal
# keywords (e.g. "took forever to arrive" won't match a "shipping" search
# unless a shared word like "arrive"/"delivery" happens to be present).
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-3.5-flash"  # the only model confirmed working on this account as of testing — gemini-2.5-flash returned 404 (retired), and model aliases like "-latest" carry the same retirement risk. Accept the tighter free-tier quota as a known tradeoff rather than chasing another model name.
MAX_RETRIEVED_REVIEWS = 12  # lower = faster response, less context per answer

_SUMMARY_INTENT_WORDS = {"summary", "summarize", "overall", "general", "everything"}
_QUESTION_FILLER_WORDS = {"what", "customers", "customer", "reviews", "review", "think", "say", "people", "give", "tell"}


def _representative_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Stratified sample across Positive/Neutral/Negative, used as a
    fallback when a question has no clear keyword target (e.g. 'summarize
    everything') or matches nothing — so the AI still sees a balanced
    cross-section rather than an arbitrary slice."""
    if len(df) <= limit:
        return df
    per_bucket = max(1, limit // 3)
    parts = []
    for lbl in ["Positive", "Neutral", "Negative"]:
        subset = df[df["sentiment"] == lbl]
        if len(subset) > 0:
            parts.append(subset.sample(min(per_bucket, len(subset)), random_state=42))
    sample = pd.concat(parts) if parts else df.head(limit)
    if len(sample) < limit:
        remaining = df.drop(sample.index)
        extra_needed = limit - len(sample)
        if len(remaining) > 0:
            sample = pd.concat([sample, remaining.sample(min(extra_needed, len(remaining)), random_state=42)])
    return sample


def retrieve_relevant_reviews(df: pd.DataFrame, question: str, limit: int = MAX_RETRIEVED_REVIEWS):
    """Finds reviews relevant to a question via keyword overlap (expanded
    with aspect keyword lists when the question mentions a known aspect),
    falling back to a representative sample for broad/summary questions
    or questions that match nothing. Returns (reviews_df, method_used)."""
    q_lower = question.lower()
    raw_words = set(re.findall(r"[a-zA-ZÀ-ÿ']{3,}", q_lower))
    q_words = raw_words - _STOPWORDS - _QUESTION_FILLER_WORDS

    if (raw_words & _SUMMARY_INTENT_WORDS) or not q_words:
        return _representative_sample(df, limit), "overview"

    expanded_words = set(q_words)
    for aspect, lang_kws in ASPECT_KEYWORDS.items():
        aspect_terms = {kw.lower() for kws in lang_kws.values() for kw in kws}
        if q_words & aspect_terms or aspect.lower() in q_lower:
            expanded_words |= aspect_terms

    scored = []
    for idx, text in df["text"].items():
        text_lower = str(text).lower()
        score = sum(1 for w in expanded_words if w in text_lower)
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return _representative_sample(df, limit), "fallback (no keyword matches)"

    scored.sort(key=lambda x: x[0], reverse=True)
    top_idx = [idx for _, idx in scored[:limit]]
    return df.loc[top_idx], "keyword search"


def build_review_prompt(question: str, retrieved_df: pd.DataFrame, full_df: pd.DataFrame, history=None):
    review_lines = []
    for _, row in retrieved_df.iterrows():
        sentiment = row.get("sentiment", "")
        snippet = str(row["text"])[:200]
        review_lines.append(f"- [{sentiment}] {snippet}")
    reviews_block = "\n".join(review_lines)

    total = len(full_df)
    counts = full_df["sentiment"].value_counts()
    overview = (
        f"Dataset overview (exact totals for the FULL dataset): {total} reviews total — "
        f"{counts.get('Positive', 0)} Positive, {counts.get('Neutral', 0)} Neutral, "
        f"{counts.get('Negative', 0)} Negative."
    )

    # Reuses the exact same aspect calculation as the Dashboard/PDF, so the
    # AI can answer aspect questions with real computed stats instead of
    # guessing from a handful of retrieved reviews.
    aspect_lines = []
    for a in ASPECTS:
        col = a.lower().replace(" ", "_") + "_stars"
        if col not in full_df.columns:
            continue
        vals = full_df[col].dropna()
        if len(vals) == 0:
            continue
        buckets = vals.apply(star_to_bucket)
        label = plurality_sentiment_label(buckets)
        bcounts = buckets.value_counts()
        pos_pct = bcounts.get("Positive", 0) / len(vals) * 100
        neg_pct = bcounts.get("Negative", 0) / len(vals) * 100
        neu_pct = bcounts.get("Neutral", 0) / len(vals) * 100
        aspect_lines.append(
            f"- {a}: {label} ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative, "
            f"{neu_pct:.0f}% neutral; mentioned in {len(vals)} of {total} reviews)"
        )
    aspect_block = (
        "Aspect breakdown (exact stats for the FULL dataset):\n" + "\n".join(aspect_lines)
        if aspect_lines else ""
    )

    # Reuses the exact same keyword extraction as the Dashboard charts.
    pos_keywords = extract_top_keywords(_english_only_text(full_df, "Positive"), top_n=8)
    neg_keywords = extract_top_keywords(_english_only_text(full_df, "Negative"), top_n=8)
    keyword_block = ""
    if pos_keywords:
        keyword_block += "Top words in Positive reviews (exact counts): " + \
            ", ".join(f"{w} ({c})" for w, c in pos_keywords) + "\n"
    if neg_keywords:
        keyword_block += "Top words in Negative reviews (exact counts): " + \
            ", ".join(f"{w} ({c})" for w, c in neg_keywords)

    system_prompt = (
        "You are a review analysis assistant. You are given several things: "
        "(1) a dataset overview with EXACT totals for the whole dataset, "
        "(2) an EXACT aspect breakdown (price, quality, delivery, etc.) computed "
        "from the full dataset — use this for any aspect-related question instead "
        "of guessing from individual reviews, since it's precise, not a sample, "
        "(3) EXACT top-keyword counts from positive and negative reviews — use "
        "this for 'what do customers like/complain about' style questions, and "
        "(4) a sample of individual reviews relevant to the question, for specific "
        "examples or quotes — this sample is NOT the full dataset, so never treat "
        "its size as the total review count.\n\n"
        "LENGTH RULES (follow these strictly, they are not suggestions):\n"
        "- If the question asks about ONE specific aspect or topic, you may go "
        "into detail: specific counts, quoted excerpts, multiple sentences.\n"
        "- If the question is broad and covers MULTIPLE aspects/topics at once "
        "(e.g. 'what can you tell me from the analysis', 'summarize everything'), "
        "give EACH aspect/topic no more than ONE short sentence — just the "
        "sentiment and one word on why. Do NOT quote specific keyword counts or "
        "review excerpts for every aspect in a broad answer; save that level of "
        "detail for when the user asks about that aspect specifically. A broad "
        "answer covering six aspects should read like six short bullet points, "
        "not six short paragraphs.\n"
        "- Total length should rarely exceed 200 words even for broad questions. "
        "A short, complete answer that invites a follow-up question is better "
        "than a long one that runs out of room.\n\n"
        "ALWAYS finish your answer as a complete thought; never trail off "
        "mid-sentence or mid-list. If none of the provided data is enough to "
        "answer, say so honestly rather than guessing or inventing details. The "
        "user may ask follow-up questions that refer back to earlier parts of "
        "the conversation — use the conversation history below for that context."
    )

    convo_block = ""
    if history:
        recent = history[-12:]  # last ~6 exchanges — more memory, still bounded so the prompt doesn't grow unbounded
        convo_lines = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent]
        convo_block = "Previous conversation:\n" + "\n".join(convo_lines) + "\n\n"

    user_prompt = (
        f"{convo_block}{overview}\n\n"
        f"{aspect_block}\n\n"
        f"{keyword_block}\n\n"
        f"Sample of {len(retrieved_df)} relevant reviews (not the full dataset):\n{reviews_block}\n\n"
        f"Question: {question}"
    )
    return system_prompt, user_prompt


def ask_ai_about_reviews_stream(full_df: pd.DataFrame, question: str, history=None):
    """Generator yielding the AI's response as accumulated text, chunk by
    chunk, so the UI can show it appearing progressively (like ChatGPT/
    Copilot) instead of waiting silently for the full response."""
    if full_df is None or len(full_df) == 0:
        yield "Run an analysis first, then ask a question about the results."
        return
    if not question or not question.strip():
        yield "Type a question first."
        return

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        yield (
            "**This feature needs a Gemini API key** (free within Google's quota "
            "limits — no credit card required).\n\n"
            "1. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).\n"
            "2. Set it as an environment variable before launching the app, e.g. "
            "`export GEMINI_API_KEY=...` in your terminal, or add it to your Hugging "
            "Face Space's secrets if deployed there."
        )
        return

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        yield "The `google-genai` package isn't installed. Run `pip install google-genai` and try again."
        return

    retrieved, _method = retrieve_relevant_reviews(full_df, question, limit=MAX_RETRIEVED_REVIEWS)
    system_prompt, user_prompt = build_review_prompt(question, retrieved, full_df, history=history)

    try:
        client = genai.Client(api_key=api_key)
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=900,
            ),
        )
        accumulated = ""
        got_any_chunk = False
        finish_reason = None
        for chunk in stream:
            if getattr(chunk, "text", None):
                accumulated += chunk.text
                got_any_chunk = True
                yield accumulated
            # Defensive check: if the SDK exposes a finish reason on this
            # chunk, remember it. Wrapped in getattr/try so this can't
            # crash if the attribute doesn't exist on some SDK version.
            try:
                candidates = getattr(chunk, "candidates", None)
                if candidates:
                    reason = getattr(candidates[0], "finish_reason", None)
                    if reason:
                        finish_reason = str(reason)
            except Exception:
                pass

        if not got_any_chunk:
            yield "(No response text was returned.)"
        elif finish_reason and finish_reason.upper() not in ("STOP", "FINISH_REASON_STOP", "1"):
            # Something other than a normal, complete finish happened
            # (e.g. a safety filter, a dropped connection, hitting the
            # token limit) — say so explicitly rather than silently
            # showing a partial answer as if it were the whole thing.
            yield accumulated + f"\n\n*(Response ended early: {finish_reason}. Try asking again.)*"
    except Exception as e:
        error_text = str(e)
        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            yield (
                "**You've hit today's free-tier limit for this model.** Google's "
                "free Gemini quota resets at midnight Pacific time — try again "
                "after that, or switch `GEMINI_MODEL` in `app.py` to a different "
                "model if this happens often."
            )
        else:
            yield f"Couldn't get a response from Gemini: {e}"


def ask_ai_about_reviews(full_df: pd.DataFrame, question: str, history=None) -> str:
    """Non-streaming convenience wrapper — returns just the final text.
    Kept in case anything needs a single return value instead of a stream."""
    final = ""
    for chunk in ask_ai_about_reviews_stream(full_df, question, history=history):
        final = chunk
    return final


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def on_mode_change(mode):
    return (
        gr.update(visible=mode == "Single text"),
        gr.update(visible=mode == "Multiple reviews (one per line)"),
        gr.update(visible=mode == "Upload file"),
        gr.update(visible=mode == "Upload file"),
    )


def on_analyze(mode, single_text, multi_text, uploaded_file, progress=gr.Progress()):
    (
        full_display_df, stat_cards_html, pie_fig, count_fig, aspect_fig,
        pos_keywords_fig, neg_keywords_fig, trend_line_fig,
        summary_text, full_df, _,
    ) = run_batch_analysis(mode, single_text, multi_text, uploaded_file, progress=progress)

    page_df, page_status, page = paginate_table(full_display_df, "", "All", "None", 1)

    all_figs = {
        "pie": pie_fig, "count": count_fig, "aspect": aspect_fig,
        "pos_keywords": pos_keywords_fig, "neg_keywords": neg_keywords_fig,
    }

    return (
        page_df, page_status, page,
        gr.update(value=""), gr.update(value="All"), gr.update(value="None"),
        stat_cards_html, pie_fig, count_fig, aspect_fig,
        pos_keywords_fig, neg_keywords_fig, trend_line_fig,
        summary_text, full_display_df, full_df, all_figs,
    )


def on_filter_change(full_display_df, search_text, sentiment_filter, sort_order):
    """Search/filter/sort changed — reset to page 1."""
    return paginate_table(full_display_df, search_text, sentiment_filter, sort_order, 1)


def on_prev_page(full_display_df, search_text, sentiment_filter, sort_order, page):
    return paginate_table(full_display_df, search_text, sentiment_filter, sort_order, (page or 1) - 1)


def on_next_page(full_display_df, search_text, sentiment_filter, sort_order, page):
    return paginate_table(full_display_df, search_text, sentiment_filter, sort_order, (page or 1) + 1)


def on_download_csv(full_df):
    path = generate_csv_report(full_df)
    if path is None:
        raise gr.Error("Run an analysis first.")
    return path


def on_download_pdf(full_df, figs):
    if full_df is None or not figs:
        raise gr.Error("Run an analysis first.")
    return generate_pdf_report(full_df, figs)


def on_ask_ai_message(full_df, message, history):
    """Handles one turn of the Ask AI chat as a stream: shows the user's
    message immediately, then yields the chat history again each time a
    new chunk of the AI's answer arrives, so the response appears
    progressively (like ChatGPT/Copilot) instead of all at once after a
    long silent wait.

    `history` is a list of {"role": ..., "content": ...} dicts — the
    format modern Gradio's Chatbot requires (newer Gradio versions dropped
    the old [user, bot] tuple-pair format entirely, so this is the only
    format that works now)."""
    history = history or []
    if not message or not message.strip():
        yield history, ""
        return

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield new_history, ""  # show the user's message + clear the input box right away

    for accumulated_text in ask_ai_about_reviews_stream(full_df, message, history=history):
        new_history[-1]["content"] = accumulated_text
        yield new_history, ""


def on_clear_chat():
    return [], ""


with gr.Blocks(title="AI-powered Review Analytics Dashboard") as demo:
    gr.Markdown(
        """
        # AI-powered Review Analytics Dashboard
        Analyze a single piece of text, paste multiple reviews, or upload a file —
        then explore the results in the dashboard and download a report.

        **How it decides Positive/Neutral/Negative:** an AI model reads each
        review and estimates how positive or negative it sounds, then that
        gets turned into a simple label. Each result also shows a
        **confidence score** — how sure the AI was about that particular
        rating. A review like "amazing, love it!" gets high confidence;
        something more mixed or vague gets lower confidence.

        Built for **English-language reviews**. Non-English reviews are
        automatically translated to English when a `GEMINI_API_KEY` is
        configured (original text is preserved alongside the translation) —
        without a key, non-English text still gets a best-effort rating, but
        keyword extraction, aspect detection, and the AI summary won't work
        reliably on it.
        """
    )

    full_results_state = gr.State(None)
    full_display_state = gr.State(None)
    all_figs_state = gr.State({})
    page_state = gr.State(1)

    with gr.Tabs():
        with gr.Tab("Analyze"):
            mode = gr.Radio(
                ["Single text", "Multiple reviews (one per line)", "Upload file"],
                value="Single text",
                label="Input method",
            )

            single_text = gr.Textbox(
                label="Text to analyze", lines=5, visible=True,
                placeholder="e.g. 'The staff were friendly but the wait was too long.'",
            )
            multi_text = gr.Textbox(
                label="Paste reviews (one per line)", lines=8, visible=False,
                placeholder="Loved the product!\nShipping took forever.\nIt's okay, nothing special.",
            )
            uploaded_file = gr.File(
                label="Upload a file",
                visible=False,
                file_types=[".csv", ".txt", ".xlsx", ".docx", ".pdf", ".json", ".png", ".jpg", ".jpeg"],
            )
            file_format_help = gr.Markdown(
                """
                **Supported file types:** .csv, .xlsx, .txt, .docx, .pdf, .json,
                or images (.png/.jpg/.jpeg). Review text is detected automatically —
                no specific column names or file structure required.
                """,
                visible=False,
            )

            mode.change(
                on_mode_change,
                inputs=mode,
                outputs=[single_text, multi_text, uploaded_file, file_format_help],
            )

            analyze_btn = gr.Button("Analyze", variant="primary")
            stat_cards_box = gr.HTML()
            summary_box = gr.Markdown()

            with gr.Row():
                search_box = gr.Textbox(label="Search reviews", placeholder="Search review text...", scale=2)
                sentiment_filter = gr.Dropdown(
                    ["All", "Positive", "Neutral", "Negative"], value="All",
                    label="Filter by sentiment", scale=1,
                )
                sort_dropdown = gr.Dropdown(
                    ["None", "Newest first", "Oldest first"], value="None",
                    label="Sort by date", scale=1,
                )

            results_table = gr.Dataframe(
                label="Results", interactive=False, wrap=True,
                value=pd.DataFrame(columns=["text", "original_text", "sentiment", "confidence", "date", "language"]),
            )

            with gr.Row():
                prev_btn = gr.Button("← Previous", size="sm")
                page_status_box = gr.Markdown("No results yet.")
                next_btn = gr.Button("Next →", size="sm")

        with gr.Tab("Dashboard"):
            gr.Markdown("Charts update automatically after you run an analysis in the **Analyze** tab.")
            with gr.Row():
                pie_plot = gr.Plot(label="Sentiment breakdown (%)")
                count_plot = gr.Plot(label="Sentiment breakdown (count)")
            gr.Markdown(
                "**Sentiment by aspect** — detected by keyword matching (price, quality, "
                "customer service, delivery, functionality, usability), not a dedicated "
                "aspect model. It catches direct mentions but may miss indirect phrasing."
            )
            aspect_plot = gr.Plot(label="Sentiment by aspect")

            trend_line_plot = gr.Plot(label="Sentiment Trend Over Time")

            gr.Markdown("Most frequently mentioned keywords in positive and negative reviews.")
            with gr.Row():
                pos_keywords_plot = gr.Plot(label="Top keywords — Positive")
                neg_keywords_plot = gr.Plot(label="Top keywords — Negative")

        with gr.Tab("Reports"):
            gr.Markdown("Download the full results as CSV, or a formatted summary as PDF.")
            with gr.Row():
                csv_btn = gr.Button("Download CSV")
                pdf_btn = gr.Button("Download PDF")
            csv_file = gr.File(label="CSV report", interactive=False)
            pdf_file = gr.File(label="PDF report", interactive=False)

        with gr.Tab("💬 Ask AI"):
            gr.Markdown(
                """
                ### Ask AI About These Reviews

                Chat with an AI about your review data — ask follow-up questions
                just like a normal conversation. Runs on Google Gemini's free API
                tier, so answers are kept fairly short and requests are rate-limited
                (see README for setup).
                """
            )

            chatbot = gr.Chatbot(label="Conversation", height=400)

            with gr.Row():
                question_box = gr.Textbox(
                    label="Ask a question", scale=4,
                    placeholder="What do customers complain about most? What do people like? Has sentiment improved over time?",
                )
                ask_ai_btn = gr.Button("Ask AI", variant="primary", scale=1)
            clear_chat_btn = gr.Button("Clear conversation", size="sm")

    analyze_btn.click(
        fn=on_analyze,
        inputs=[mode, single_text, multi_text, uploaded_file],
        outputs=[
            results_table, page_status_box, page_state,
            search_box, sentiment_filter, sort_dropdown,
            stat_cards_box, pie_plot, count_plot, aspect_plot,
            pos_keywords_plot, neg_keywords_plot, trend_line_plot,
            summary_box, full_display_state, full_results_state, all_figs_state,
        ],
    )

    _filter_inputs = [full_display_state, search_box, sentiment_filter, sort_dropdown]
    _table_outputs = [results_table, page_status_box, page_state]

    search_box.change(fn=on_filter_change, inputs=_filter_inputs, outputs=_table_outputs)
    sentiment_filter.change(fn=on_filter_change, inputs=_filter_inputs, outputs=_table_outputs)
    sort_dropdown.change(fn=on_filter_change, inputs=_filter_inputs, outputs=_table_outputs)
    prev_btn.click(fn=on_prev_page, inputs=_filter_inputs + [page_state], outputs=_table_outputs)
    next_btn.click(fn=on_next_page, inputs=_filter_inputs + [page_state], outputs=_table_outputs)

    csv_btn.click(fn=on_download_csv, inputs=full_results_state, outputs=csv_file)
    pdf_btn.click(fn=on_download_pdf, inputs=[full_results_state, all_figs_state], outputs=pdf_file)

    _chat_inputs = [full_results_state, question_box, chatbot]
    _chat_outputs = [chatbot, question_box]

    ask_ai_btn.click(fn=on_ask_ai_message, inputs=_chat_inputs, outputs=_chat_outputs)
    question_box.submit(fn=on_ask_ai_message, inputs=_chat_inputs, outputs=_chat_outputs)
    clear_chat_btn.click(fn=on_clear_chat, inputs=None, outputs=[chatbot, question_box])

if __name__ == "__main__":
    # Render (and most hosting platforms) assign a port via the PORT
    # environment variable and require binding to 0.0.0.0, not just
    # localhost. Locally, PORT won't be set, so this falls back to
    # Gradio's normal default (7860) — no change for local use.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
