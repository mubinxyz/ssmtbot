import sys
import os

# ensure project root / current dir is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

# import the Flask app object (exposed as `app` in wsgi_bot.py)
from wsgi_bot import app as application
