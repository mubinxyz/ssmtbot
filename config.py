# config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is required")

# Read host part from env (no scheme, no path)
WEBHOOKS_HOST = os.environ.get("WEBHOOKS_URL")
if not WEBHOOKS_HOST:
    raise ValueError("WEBHOOKS_URL environment variable is required (e.g. 'yourdomain.example')")

# Normalize: strip whitespace and any scheme/path the user might have mistakenly provided
WEBHOOKS_HOST = WEBHOOKS_HOST.strip().rstrip("/")  # remove trailing slash if any
# if someone included a scheme, remove it:
if WEBHOOKS_HOST.startswith("http://"):
    WEBHOOKS_HOST = WEBHOOKS_HOST[len("http://"):]
elif WEBHOOKS_HOST.startswith("https://"):
    WEBHOOKS_HOST = WEBHOOKS_HOST[len("https://"):]

# Build full webhook URL that Telegram expects (include token path)
WEBHOOK_URL = f"https://{WEBHOOKS_HOST}/webhook/{BOT_TOKEN}"

# Logging level (convert friendly name to logging constant)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, _LOG_LEVEL, logging.INFO)

# DB path and other envs
DB_PATH = os.getenv("DB_PATH", "database.db")
