# download_model.py
from sentence_transformers import SentenceTransformer

# This line downloads the model from the internet.
# Run this script once locally to save the model files.
# The Docker image will then use these local files.
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('./model')
print("Model downloaded and saved to ./model directory.")