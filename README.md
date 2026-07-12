# AI-powered Review Analytics Dashboard (Python + Gradio)

A batch-capable sentiment analysis tool for **English-language reviews**,
with star ratings, trend analysis, a visualization dashboard, aspect-based
breakdowns, and downloadable reports — built on
`tabularisai/multilingual-sentiment-analysis`, a pre-trained transformer
model (used here for English text only).

## Features

1. **Multiple input options** — analyze a single piece of text, paste many
   reviews at once (one per line), or upload a file in any of these formats:
   `.csv`, `.txt`, `.xlsx`, `.docx`, `.pdf`, `.json`, or images (`.png`,
   `.jpg`, `.jpeg`) — text is pulled out of images via OCR.
2. **Star ratings (1-5) as the main result**, with Positive/Neutral/Negative
   kept as a smaller secondary label. Stars are a weighted average across
   the model's 5 internal confidence levels (not just whichever one "wins"),
   so a review can land at 4.3 stars rather than being forced into a flat 4.
3. **Time-based trend analysis** — if your uploaded data has a `date` column
   (csv/xlsx/json only), the trend chart plots average star rating by date.
   Without one, it falls back to plotting by review order and says so explicitly.
4. **Visualization dashboard** — a star-distribution bar chart (main), a
   Positive/Neutral/Negative pie chart (secondary view of the same data),
   a sentiment trend line, and an aspect breakdown chart, all in one tab.
5. **Batch processing** — every input mode runs through the same pipeline
   and produces one results table.
6. **Downloadable reports** — a CSV of the full results, and a formatted
   PDF summary (average star rating, breakdown, aspect summary, highest/
   lowest-rated examples, and the star distribution chart).
7. **Aspect-based sentiment** — for each review, a star rating is broken
   down by **Price, Quality, Customer Service, Delivery, Functionality,
   and Usability** wherever those topics are mentioned.
8. **Searchable, filterable, paginated results table** — search review
   text, filter by sentiment, sort by date, browse in pages of 15.
9. **Structured AI summary** — overall sentiment plus auto-generated
   Strengths/Issues bullets, pulled from the top-keyword extraction.
10. **Summary stat cards** — Total/Positive/Negative/Neutral counts at a glance.
11. **"Ask AI" chat about your reviews** — a simple RAG pipeline (keyword
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

## Ask AI About These Reviews — free within Google's quota

This feature calls Google's **Gemini API** to answer questions about your
reviews. Unlike OpenAI's API, Gemini has a genuine free tier — no credit
card required, rate-limited (roughly 15 requests/minute and 1,500/day for
Flash models as of this writing) but free for typical demo/portfolio use.
Very heavy sustained usage could eventually hit that quota and need billing
enabled, but normal use should stay free — consistent with the rest of
this app.

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
4. Up to 25 retrieved reviews (change `MAX_RETRIEVED_REVIEWS` in `app.py`)
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

## Deploy it for free

Push `app.py`, `requirements.txt`, and `packages.txt` to a Hugging Face
Space with the Gradio SDK, and it builds and hosts automatically on the
free CPU tier. The "Ask AI" tab needs its own `GEMINI_API_KEY` secret set
on the Space (see the section above) — free within Google's quota, same
as running locally.

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

- **Built for English-language reviews.** The sentiment model itself is
  multilingual under the hood, so non-English text will still get some
  rating, but keyword extraction, aspect detection, and the AI summary are
  all English-only and won't work reliably on other languages.
- **PDF and DOCX uploads treat every line/paragraph as one review.** If your
  document has reviews spanning multiple lines each, they'll get split up
  incorrectly — these formats work best for genuinely line-per-review data.
- **Image OCR accuracy depends heavily on image quality.** Clear, typed
  text on a plain background works well; handwriting, low resolution, or
  skewed/angled photos often extract poorly or produce garbled text.
- **Aspect analysis makes batch processing slower**, since each mentioned
  aspect triggers an additional model call per review (up to 6 extra per
  review, on top of the main sentiment call). Large batches on CPU-only
  hosting (like the free Hugging Face Spaces tier) will take noticeably
  longer than before this feature was added.
- Free-tier hosting (Hugging Face Spaces) can take a few seconds to wake
  up after inactivity — expected behavior, not a bug.
