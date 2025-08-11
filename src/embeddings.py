import face_recognition
import cv2

# ---- Helper: get 128D embedding ----
def get_embedding(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img,face_locations)
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None
    
# ---- Helper : add padding ------
def pad_crop(frame, x1, y1, x2, y2, padding=20):
    height, width = frame.shape[:2]

    # Expand the box with padding, but clip to image boundaries
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(width, x2 + padding)
    y2_p = min(height, y2 + padding)

    return frame[y1_p:y2_p, x1_p:x2_p]


import json
import numpy as np

# Load as dict with list values
db_path = "db/embeddings_db.json"

def load_db(path):
    known_f = {}
    with open(path, "r") as f:
        db = json.load(f)
    for name,embed in db.items():
        known_f[name] = np.array(embed)
    # print(f"Total person found in DB : {known_f.keys()}")
    return known_f

