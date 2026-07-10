# Week 2 — LLM API Mini-Projects

Three small scripts built while learning to work with the Anthropic (Claude) API.
The goal of this week was to get hands-on with the core building blocks of an LLM app:
prompting, structured/constrained output, conversation memory, and basic API error handling.

## Projects

- **`summarize_scripts.py`** — Summarizes a block of text into a short, controlled output
  (fixed length, key points only) using a system prompt.
- **`classify_scripts.py`** — Classifies text (e.g. sentiment) into a fixed set of labels
  using few-shot examples and output constraints.
- **`chat_history_rewrite.py`** — A Q&A chatbot that remembers context by resending the
  running conversation history on each turn.
- **`index.py`** - The refactor version includes the three mini project.

## Stack

- Python 3
- Anthropic Python SDK (`anthropic`)
- `python-dotenv` for managing the API key

## Setup

```bash
pip install anthropic python-dotenv
cp .env.example .env      # then add your ANTHROPIC_API_KEY
```

> `.env` is git-ignored — only `.env.example` is committed.

## Notes

Part of a 20-week AI Engineer learning journey. These are learning exercises meant to
demonstrate the fundamentals rather than production-ready tools.
