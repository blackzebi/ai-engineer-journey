"""Sanity-check that the local venv can reach the Anthropic API using a key from .env.

Usage:
    .venv\\Scripts\\python.exe scripts\\test_api_call.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

from anthropic import Anthropic

client = Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=64,
    messages=[{"role": "user", "content": "Reply with the single word: pong"}],
)

print(response.content[0].text)
