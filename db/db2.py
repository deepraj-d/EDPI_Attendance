import os
import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use CUDAExecutionProvider if GPU available
app.prepare(ctx_id=0, det_size=(640, 640)) 

# Storage for embeddings
db = {}

# Directory containing reference face images
folder_name = 'refrence_images'

# Function to extract 512-D ArcFace embedding
def get_arcface_embedding(image):
    faces = app.get(image)
    if faces:
        # Return the embedding of the most prominent face
        return faces[0].embedding
    return None

# Loop through each image
for filename in os.listdir(folder_name):
    if filename.startswith("."):
        continue  # Skip hidden files

    path = os.path.join(folder_name, filename)
    image = cv2.imread(path)
    name = os.path.splitext(filename)[0]

    embedding = get_arcface_embedding(image)
    print(f"Processed: {filename}")

    if embedding is not None:
        db[name] = embedding.tolist()
    else:
        print(f"\033[31mCould not extract embedding for {filename}\033[0m")

# Save to JSON file
os.makedirs("db", exist_ok=True)
with open("db/embeddings_db.json", "w") as f:
    json.dump(db, f)

print("\n✅ All embeddings saved to db/embeddings_db.json")
