# AI-powered Review Analytics Dashboard

*A personal portfolio project demonstrating end-to-end AI application
development — local ML inference, an AI-powered chat feature with
retrieval, data visualization, and automated reporting.*

🔗 **[Try it live](https://junio7191-sentiment-dashboard.hf.space)**

---

## Project Overview

A batch-capable sentiment analysis tool for English-language customer
reviews. Upload a single review, paste many at once, or upload a file —
the app classifies each as Positive, Neutral, or Negative, breaks
sentiment down by aspect (price, quality, delivery, etc.), and lets you
chat with an AI about the results. Non-English reviews are automatically
translated so the whole pipeline works on them too.

Built to explore a full AI application stack in one project: a local
transformer model for the core task, a cloud LLM for a conversational
feature layered on top, and the practical realities of shipping both —
free-tier limits, model retirements, and hosting constraints included.

## Live Demo

👉 **[junio7191-sentiment-dashboard.hf.space](https://junio7191-sentiment-dashboard.hf.space)**

No installation needed — upload a review file and try it directly.

## Features

- **Sentiment classification** (Positive/Neutral/Negative) with a
  confidence score per review
- **Aspect-based breakdown** — Price, Quality, Customer Service,
  Delivery, Functionality, Usability
- **Automatic translation** of non-English reviews (original text
  preserved alongside)
- **"Ask AI" chat** — ask natural-language questions about your data,
  grounded in exact computed statistics, not guesses
- **Visual dashboard** — sentiment charts, weekly trend line, top
  keyword extraction
- **File support** — `.csv`, `.xlsx`, `.txt`, `.docx`, `.pdf`, `.json`,
  and images via OCR
- **Downloadable reports** — full CSV export and a formatted PDF summary
- **Searchable, paginated results table**

## Tech Stack

| Layer | Technology |
|---|---|
| UI / App framework | [Gradio](https://gradio.app) |
| Sentiment model | `tabularisai/multilingual-sentiment-analysis` (Hugging Face Transformers, runs locally) |
| Conversational AI | Google Gemini API (`gemini-3.5-flash`) |
| Data processing | pandas |
| Visualization | Matplotlib, Plotly |
| Document generation | fpdf2 (PDF), openpyxl (Excel) |
| OCR | Tesseract (via pytesseract) |
| Language detection | langdetect |
| Hosting | Hugging Face Spaces |

## Architecture

```
Upload (text / file / image)
        │
        ▼
 Language detection (local)
        │
   non-English? ──yes──► Gemini translation (batched)
        │                        │
        ▼                        ▼
 Local sentiment model ◄─────────┘
 (English text only)
        │
        ▼
 Aspect detection + keyword extraction
        │
        ▼
 Dashboard charts · CSV/PDF export · Ask AI chat
 (Ask AI also pulls precomputed aspect/keyword
  stats directly, not just raw review samples)
```

The core analysis pipeline is fully local and free — no API required.
Translation and the Ask AI chat are the only features that call an
external API (Gemini), and both degrade gracefully without one.

## Screenshots

*(Add screenshots here — e.g. the Analyze tab, Dashboard charts, and
Ask AI conversation. Drag image files into this section on GitHub, or
reference them like `![Dashboard](screenshots/dashboard.png)`.)*

## How to Run

**Option 1 — use the live demo above, no setup needed.**

**Option 2 — run it locally:**
```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python app.py
```
Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

For the "Ask AI" chat and automatic translation, set a free Gemini API
key (get one at [Google AI Studio](https://aistudio.google.com/apikey)):
```bash
export GEMINI_API_KEY=your-key-here
```
Image upload (OCR) needs Tesseract installed separately — see comments
in `app.py` for platform-specific install commands.

## Known Limitations

- **Aspect detection is keyword-based**, not a fine-tuned model — it
  catches direct mentions but can miss indirect phrasing.
- **"Ask AI" runs on Gemini's free tier**, which has a limited daily
  request quota. It may be briefly unavailable if that's reached; every
  other feature is unaffected.
- **Translation requires an API key** — without one, non-English reviews
  still get a best-effort rating, but aspect detection and keyword
  extraction won't work reliably on them.
- **Ask AI's retrieval is keyword matching**, not semantic search — it
  can miss reviews that are relevant in meaning without sharing literal
  words.
- **PDF/DOCX uploads treat each line/paragraph as one review** — best
  suited to genuinely one-review-per-line data.

## Future Improvements

- Replace keyword-based aspect detection with a fine-tuned classifier
- Real semantic search (embeddings + vector store) for Ask AI's
  retrieval, instead of keyword matching
- Expand beyond English as the primary analysis language
- Support for additional file formats and batch upload of multiple files
