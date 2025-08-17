# config.py
import os
from dotenv import load_dotenv
import logging

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
LOG_LEVEL = os.getenv('LOG_LEVEL')
DB_PATH = os.getenv("DB_PATH", "database.db")


