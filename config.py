"""Application configuration for Smart PDF Summarizer.

This file centralizes Flask and MongoDB settings so the app stays simple to
understand and easy to deploy in a college project setting.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# Load project-local .env before reading environment-backed settings.
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base Flask configuration loaded from environment variables."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/smart_pdf")
    UPLOAD_FOLDER = "/tmp" if os.environ.get("VERCEL") else str(BASE_DIR / "static" / "uploads")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    JSON_SORT_KEYS = False


class DevelopmentConfig(Config):
    """Development-specific settings."""

    DEBUG = True


class ProductionConfig(Config):
    """Production-specific settings."""

    DEBUG = False
