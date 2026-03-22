"""WSGI entrypoint: gunicorn wsgi:app (run from webapp/)."""
from flask_app import create_app

app = create_app()
