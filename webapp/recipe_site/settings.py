"""Django settings for the Recipe Traffic Predictor web app."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # webapp/
PROJECT_DIR = BASE_DIR.parent                           # repository root

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "recipe_site.urls"
WSGI_APPLICATION = "recipe_site.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    },
]

# The app has no user accounts or stored records, so no database is needed.
DATABASES = {}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
# Expose the analysis figures from the repository's images/ folder
STATICFILES_DIRS = [("figures", PROJECT_DIR / "images")]

DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024

# Machine learning
RECIPE_DATA_PATH = PROJECT_DIR / "data" / "recipe_site_traffic_2212.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
