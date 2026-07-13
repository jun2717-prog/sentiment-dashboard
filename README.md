---
title: AI-powered Review Analytics Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.37.0"
app_file: app.py
pinned: false
---

# AI-powered Review Analytics Dashboard (Python + Gradio)

A batch-capable sentiment analysis tool for **English-language reviews**
(with automatic translation for non-English reviews when a Gemini API key
is configured), with Positive/Neutral/Negative classification, trend
analysis, a visualization dashboard, aspect-based breakdowns, and
downloadable reports — built on `tabularisai/multilingual-sentiment-analysis`,
a pre-trained transformer model.

## Features

1. **Multiple input options** — analyze a single piece of text, paste many
   reviews at once (one per line), or upload a file in any of these formats:
   `.csv`, `.txt`, `.xlsx`, `.docx`, `.pdf`, `.json`, or images (`.png`,
   `.jpg`, `.jpeg`) — text is pulled out of images via OCR.
2. **Language detection with automatic translation** — non-English reviews
   are detected locally (free) and translated to English via Gemini when
   `GEMINI_API_KEY` is set, so the rest of the pipeline works reliably on
   them. Original text is always preserved alongside the translation — see
   "Automatic translation of non-English reviews" below for the full
   tradeoffs (this is the one part of the app that needs an API key).
3. **Positive/Neutral/Negative classification**, with a per-item confidence
   score showing how sure the model was — internally, this is derived from
   a weighted average across the model's 5 confidence levels (not just
   whichever one "wins"), then mapped to one of the three labels. No
   numeric star rating is shown anywhere in the UI.
4. **Time-based trend analysis** — if your uploaded data has a `date` column
   (csv/xlsx/json only), the trend chart plots sentiment counts by week.
   Without one, it shows a clear "no date data" message instead.
5. **Visualization dashboard** — a Positive/Neutral/Negative pie chart, a
   matching count bar chart, a weekly sentiment trend line, an aspect
   breakdown chart, and top-keyword charts for positive/negative reviews.
6. **Batch processing** — every input mode runs through the same pipeline
   and produces one results table.
7. **Downloadable reports** — a CSV of the full results, and a formatted
   PDF summary (overall sentiment, Positive/Neutral/Negative breakdown,
   all five dashboard charts, aspect summary, and example positive/negative
   reviews). Both use Positive/Neutral/Negative only, no numeric scores.
8. **Aspect-based sentiment** — for each review, sentiment is broken
   down by **Price, Quality, Customer Service, Delivery, Functionality,
   and Usability** wherever those topics are mentioned.
9. **Searchable, filterable, paginated results table** — search review
   text, filter by sentiment, sort by date, browse in pages of 15.
10. **Structured AI summary** — overall sentiment plus auto-generated
    Strengths/Issues bullets, pulled from the top-keyword extraction.
11. **Summary stat cards** — Total/Positive/Negative/Neutral counts at a glance.
12. **"Ask AI" chat about your reviews** — a simple RAG pipeline (keyword
    retrieval, not a vector database) feeding into Google's Gemini API,
    which has a free tier — see the dedicated section below for setup.

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

## Supported upload formats

| Format  | How reviews are identified |
|---------|------------------------------|
| `.csv`  | Looks for a column named `text`/`review`/`reviews`/`comment` (case-insensitive), else uses the first column. An optional `date`/`timestamp`/`time` column enables the trend chart. |
| `.xlsx` | Same column detection as `.csv`. |
| `.txt`  | Each non-empty line is treated as one review. |
| `.docx` | Each non-empty paragraph is treated as one review. |
| `.pdf`  | Text is extracted from every page; each non-empty line is treated as one review. |
| `.png` / `.jpg` / `.jpeg` | Text is extracted via OCR (Tesseract, English), each non-empty line is treated as one review. Works best on clear, high-contrast text — handwriting or low-quality photos will extract poorly or not at all. |
| `.json` | Accepts a list of strings, a list of `{"text": ..., "date": ...}` objects, or a dict wrapping either of those under any key (e.g. `{"reviews": [...]}`). |

Example CSV with a date column:

```csv
text,date
"Loved the product, fast shipping!",2026-01-03
"Customer service never responded.",2026-01-04
"It's fine, does what it says.",2026-01-05
```

## Image upload (OCR) needs one extra install step

Unlike every other format, reading text out of images requires a program
called **Tesseract OCR** to be installed on the machine itself — installing
the `pytesseract` Python package alone is not enough, since it's just a
wrapper around that program.

- **Mac**: `brew install tesseract`
- **Ubuntu/Debian/Linux**: `sudo apt-get install tesseract-ocr`
- **Windows**: download the installer from the
  [Tesseract project's UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
  and add it to your system PATH.

If it's missing, the app will show a clear message asking for this install
rather than a generic error. If you deploy to **Hugging Face Spaces**, add
the included `packages.txt` file alongside `app.py` and `requirements.txt` —
Spaces reads it automatically to install system-level packages like this one.

## Automatic translation of non-English reviews

If your uploaded data includes non-English reviews, they're automatically
translated to English (via the same Gemini API used by Ask AI) so that
sentiment analysis, aspect detection, keyword extraction, and the AI
summary all work reliably on them — those parts of the app are tuned for
English and don't work well on other languages directly.

**Important tradeoff, worth knowing before you rely on this**:
- **This requires `GEMINI_API_KEY` to be set** — previously, the core
  Analyze feature was 100% local and free with zero external dependencies.
  If your data has any non-English text, that's no longer true; without
  the key, non-English reviews just get a best-effort rating from the
  (inherently multilingual) sentiment model directly, with less reliable
  aspect/keyword results.
- **It adds real time to analysis** — translation calls happen over the
  network, batched efficiently (about 20 reviews per API call rather than
  one call per review, to stay within Gemini's free-tier rate limit), but
  a dataset with many non-English reviews will take noticeably longer to
  analyze than an all-English one.
- Language is detected locally first (fast, free, via `langdetect`) —
  only reviews that are actually detected as non-English get translated,
  so an all-English dataset triggers zero API calls and stays exactly as
  fast and free as before.
- The original text is always preserved in a separate `original_text`
  column (table, CSV) — translation never destroys the original wording.
  Reviews under 15 characters are always treated as English regardless of
  what's detected, since language detection is unreliable on very short
  text and a false positive there would waste an API call translating
  something like "Great!" unnecessarily.

## Ask AI About These Reviews — free within Google's quota

This feature calls Google's **Gemini API** to answer questions about your
reviews. Unlike OpenAI's API, Gemini has a genuine free tier — no credit
card required. **The exact daily/per-minute limit is not worth quoting
here** — it's changed multiple times and varies by model and by project,
and an earlier version of this README stated a number that turned out to
be wrong for the model actually in use. If you hit a `429
RESOURCE_EXHAUSTED` error, that's a real daily quota limit, not a bug —
the chat shows a clear message when this happens, and the quota resets at
midnight Pacific time. Check your live, current limit in
[Google AI Studio](https://aistudio.google.com) rather than trusting a
number in this file. Newer/preview models tend to have tighter free
quotas than established ones — `GEMINI_MODEL` near the top of `app.py`
is the place to switch models if you're hitting limits often.

### Setup

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey)
   (no credit card needed).
2. Set it as an environment variable before launching the app:
   ```bash
   export GEMINI_API_KEY=...
   python app.py
   ```
   On Hugging Face Spaces, add `GEMINI_API_KEY` under your Space's Settings
   → Repository secrets instead of putting it in code.
3. Without this set, the tab shows a clear message with a link to get a
   key, instead of failing silently or crashing.

### How it actually works (a simple RAG pipeline, not a vector database)

1. Your question is matched against review text via **keyword overlap**
   — the same stopword list and aspect keyword lists (price, quality,
   delivery, etc.) used elsewhere in the app. If your question mentions a
   known aspect topic, the search expands to that aspect's full keyword list.
2. Broad questions ("summarize everything," "overall thoughts") skip
   keyword matching and instead pull a **stratified sample** across
   Positive/Neutral/Negative, so the AI sees a representative cross-section.
3. Questions matching nothing fall back to that same representative sample
   rather than returning an empty result.
4. Up to 12 retrieved reviews (change `MAX_RETRIEVED_REVIEWS` in `app.py`)
   are sent to `gemini-2.5-flash` (change `GEMINI_MODEL` to use a different
   model) along with your question, with instructions to answer only from
   the provided reviews.

**Why keyword search instead of a real vector database**: it's simpler,
needs no extra infrastructure (no embedding model, no vector store), and
works reasonably well at the scale this app targets (hundreds of reviews).
The real tradeoff: it can miss reviews that are relevant in meaning without
sharing literal keywords — e.g. "took forever to arrive" won't match a
"shipping" question unless a shared word like "arrive"/"delivery" is
present. For much larger datasets or more nuanced retrieval, a proper
embedding-based vector search would do better, at the cost of real
implementation complexity this version deliberately avoids.

## Deploying to Hugging Face Spaces

**Important, current-as-of-2026 cost note**: as of this writing, Hugging
Face requires a **PRO subscription ($9/month)** to host a Gradio Space on
the free CPU tier — new free accounts can only select the "Static" SDK
(plain HTML, which can't run this app) without upgrading. This is a
recent platform change; check
[huggingface.co/pricing](https://huggingface.co/pricing) for the current
state, since it may change again. Once on PRO, hosting itself has no
additional usage cost beyond the subscription for a CPU-only app like
this one.

### Steps

1. Subscribe to PRO at [huggingface.co/pro](https://huggingface.co/pro) if needed.
2. Create a new Space: profile icon → **New Space** → choose the **Gradio**
   SDK → **CPU Basic** hardware (free once on PRO) → **Blank** template.
3. Go to the Space's **Files** tab → **Add file** → **Upload files** →
   upload `app.py`, `requirements.txt`, `packages.txt`, and this
   `README.md` — **all four together**, since `app.py` alone won't build
   without the others, and `README.md` specifically won't work without
   the YAML block at the very top of this file (see below).
4. Under **Settings → Repository secrets**, add `GEMINI_API_KEY` with your
   actual key so the "Ask AI" tab works on the live version — never put
   the key directly in the code.
5. The Space builds automatically after upload — check the **"App"** tab
   for build progress and any errors.

### Why this README has a YAML block at the very top

The `---` block with `title`, `sdk`, `app_file`, etc. at the top of this
file is **not just documentation** — Hugging Face Spaces reads it to know
which SDK to use and which file to run. If you edit this README (e.g. to
customize the description) and accidentally delete that block, the Space
will show a **"Configuration error"** and won't build at all. Keep it
intact; edit the prose below it freely.

### A quick, honest gotcha we hit while setting this up

Files downloaded multiple times from a browser often get auto-renamed
(e.g. `app.py` becomes `app (1).py`). Hugging Face needs the file named
**exactly** `app.py` — double-check the filename in Finder/Explorer
before uploading, since a renamed file will build successfully but then
fail with `can't open file 'app.py': No such file or directory`.

## Aspect-based sentiment: how it actually works

For each review, the app looks for sentences that mention six aspects —
**Price, Quality, Customer Service, Delivery, Functionality, Usability** —
using English keyword lists. Whichever sentences match get run through
the sentiment model separately, so you get a per-aspect sentiment
alongside the overall one.

**This is keyword matching, not a fine-tuned aspect model.** There isn't a
practical pretrained model for arbitrary custom categories, so this is a
transparent, simpler alternative:
- It catches direct mentions: *"customer service was unhelpful"* → Customer
  Service: Negative.
- It can miss indirect ones: *"nobody got back to me for weeks"* means the
  same thing but doesn't contain any of the matched keywords, so it won't
  be attributed to Customer Service.
- If an aspect isn't mentioned at all in a review, it's marked
  **"Not mentioned"** rather than guessed at.
- The aspect chart in the Dashboard only averages reviews that actually
  mentioned that aspect (shown as "n=" on each bar), not the full batch.

If keyword coverage misses phrasing that matters for your use case, the
keyword lists near the top of `app.py` (`ASPECT_KEYWORDS`) are the place
to extend — add more synonyms per aspect.

## Top keywords: how it actually works

For each review, meaningful words (and frequent two-word phrases, like
"customer service") are extracted and counted separately for Positive and
Negative reviews, then shown as the top 12 for each.

- **Filtering**: common filler words (including subjective filler like
  "seem"/"feel"/"fine"), numbers, and overly generic verbs
  (work/use/get/make/etc.) are removed, since they don't tell you much
  about what customers actually liked or disliked.
- **Word merging**: word variants are grouped internally (e.g.
  "crashed"/"crashes"/"crashing" all count together), but the label
  actually shown is whichever original form was most common in that
  group — not an artificial stem. A group that's mostly "shipping"
  displays as "shipping"; one that's mostly "crashed"/"crashes" displays
  as "crash". This is suffix-based grouping, not true lemmatization, and
  with very small samples where forms are exactly tied the choice isn't
  meaningful — it gets more reliable as more reviews are analyzed.
- **Phrases**: adjacent word pairs that appear together at least twice
  (e.g. "customer service", "battery life") are surfaced as their own
  keyword. If a pair's two words almost always appear together (≥60% of
  each word's occurrences), the standalone words are dropped so you see
  "customer service" once instead of "customer", "service", and
  "customer service" all cluttering the list separately.

## Known limitations (worth knowing before showing this off)

- **English-language analysis, with automatic translation as a bridge.**
  Non-English reviews get translated to English automatically if
  `GEMINI_API_KEY` is set (see "Automatic translation" above), so keyword
  extraction, aspect detection, and the AI summary work reliably on them.
  Without that key, non-English text still gets a best-effort rating from
  the underlying multilingual model, but those other features won't work
  reliably on it.
- **PDF and DOCX uploads treat every line/paragraph as one review.** If your
  document has reviews spanning multiple lines each, they'll get split up
  incorrectly — these formats work best for genuinely line-per-review data.
- **Image OCR accuracy depends heavily on image quality.** Clear, typed
  text on a plain background works well; handwriting, low resolution, or
  skewed/angled photos often extract poorly or produce garbled text.
- **Aspect analysis makes batch processing slower**, since each mentioned
  aspect triggers an additional model call per review (up to 6 extra per
  review, on top of the main sentiment call). Large batches on CPU-only
  hosting will take noticeably longer than before this feature was added.
- **Hosting this on Hugging Face Spaces currently requires a PRO
  subscription** ($9/month, as of this writing) — see "Deploying to
  Hugging Face Spaces" above. This isn't a limitation of the app itself,
  but worth knowing before assuming a fully free deployment path.
- Hosted Spaces may go idle and take a few seconds to wake up after a
  period of inactivity — expected platform behavior, not a bug in this app.
