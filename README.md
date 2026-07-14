# AI-powered Review Analytics Dashboard

I built this as a personal portfolio project to go beyond a basic
sentiment classifier and put together a full AI application: a local ML
model doing the core work, a cloud LLM layered on top for a
conversational feature, and everything else — visualization, reporting,
deployment — that makes it a real, usable tool instead of just a script.

📁**[sampele file to upload](sample_reviews_50_multilingual.csv)**
🔗 **[Try it live](https://junio7191-sentiment-dashboard.hf.space)** — no installation needed


---

## Overview

Upload customer reviews — a single piece of text, a pasted list, or a
file — and get back sentiment classification, an aspect-level breakdown
(price, quality, delivery, etc.), visual charts, and an AI chat you can
ask questions about the results. Non-English reviews get translated
automatically. The core analysis runs entirely locally and for free; an
API key is only needed for translation and the chat.

## Features

- Sentiment classification (Positive / Neutral / Negative) with a
  confidence score per review
- Aspect breakdown across six categories: Price, Quality, Customer
  Service, Delivery, Functionality, Usability
- Automatic translation of non-English reviews, with the original text
  always preserved alongside
- **Ask AI** — a chat feature grounded in real computed statistics from
  the data, not just guesses from raw text
- Dashboard: sentiment charts, a weekly trend line, top keyword extraction
- Upload support for `.csv`, `.xlsx`, `.txt`, `.docx`, `.pdf`, `.json`,
  and images via OCR
- Export results to CSV or a formatted PDF report

## Tech Stack

Python · Gradio · Hugging Face Transformers (local sentiment model) ·
Google Gemini API (translation + chat) · pandas · Matplotlib/Plotly ·
Tesseract OCR · Hugging Face Spaces (hosting)

## How to Run

**Easiest way: just use the live demo above.**

To run it yourself:
```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python app.py
```
For translation and the Ask AI chat, get a free key at
[Google AI Studio](https://aistudio.google.com/apikey) and set it:
```bash
export GEMINI_API_KEY=your-key-here
```

## Downloaded csv file from the dashboard

*(Add screenshots here, e.g. `![Dashboard](screenshots/dashboard.png)`)*

## Known Limitations

I'd rather be upfront about these than let someone find them first:

- Aspect detection uses keyword matching, not a fine-tuned model — it
  catches direct mentions but can miss indirect phrasing
- Ask AI runs on Gemini's free tier, which has a daily request limit
- Translation and the chat need an API key; the core analysis doesn't
- Ask AI's retrieval is keyword-based, not true semantic search

## What I'd Improve Next

- Swap keyword-based aspect detection for a fine-tuned classifier
- Add real semantic search (embeddings) for Ask AI's retrieval
- Extend beyond English as the primary analysis language
