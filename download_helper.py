# download_helper.py
from sentence_transformers import SentenceTransformer
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
HELPER_PATH = os.path.join(MODELS_DIR, "minilm-helper")

print(f"Downloading Helper AI model to: {HELPER_PATH}")

# This is a very popular, small, and fast model for keyword extraction
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

model = SentenceTransformer(model_name)
model.save(HELPER_PATH)

print("Helper AI model downloaded successfully.")