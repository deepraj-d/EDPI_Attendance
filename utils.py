from datetime import datetime
import numpy as np
import time
import cv2

def get_employee_name_arcface(student_face, known_faces, model_app, threshold=0.3):
    """
    Detects and embeds face using InsightFace `model_app`, matches with `known_faces`,
    and returns recognized name and similarity score.

    Parameters:
        student_face (np.ndarray): Cropped face image (BGR)
        known_faces (dict): {name: embedding (list or np.array)}
        model_app (FaceAnalysis): Initialized InsightFace FaceAnalysis object
        threshold (float): Cosine similarity threshold for recognition

    Returns:
        (str or None, float or None): Recognized name and similarity score
    """
    
    # InsightFace expects RGB
    student_face_rgb = cv2.cvtColor(student_face, cv2.COLOR_BGR2RGB)

    # Run InsightFace on input
    faces = model_app.get(student_face_rgb)

    if not faces:
        return None, None

    # Use the most prominent face
    embedding = faces[0].embedding

    best_match = None
    best_score = -1

    for name, known_embedding in known_faces.items():
        known_embedding = np.array(known_embedding).reshape(1, -1)
        score = cosine_similarity([embedding], known_embedding)

        if score > best_score and score > threshold:
            best_score = score
            best_match = name

    return best_match, best_score if best_match else (None, None)



# ---- Helper: compute cosine similarity ----
def cosine_similarity(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---- Helper : add padding ------
def pad_crop(frame, x1, y1, x2, y2, padding=20):
    height, width = frame.shape[:2]

    # Expand the box with padding, but clip to image boundaries
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(width, x2 + padding)
    y2_p = min(height, y2 + padding)

    return frame[y1_p:y2_p, x1_p:x2_p]


    
def get_time(date=False, time=False):
    now = datetime.now()

    if date and time:
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif date:
        return now.strftime("%Y-%m-%d")
    elif time:
        return now.strftime("%H:%M:%S")
    else:
        return "Please specify either date or time"
    




def open_camera(url, retries=3, delay=2):
    for attempt in range(retries):
        cap = cv2.VideoCapture(url,cv2.CAP_FFMPEG)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"\033[94m[INFO] Camera FPS: {fps:.2f}\033[0m")
        if cap.isOpened():
            print("\033[92m[INFO] Camera connected.\033[0m")
            return cap
        print(f"\033[93m[WARN] Retry {attempt + 1}/{retries} - Could not open stream. Retrying in {delay}s...\033[0m")
        time.sleep(delay)
    print("\033[91m[ERROR] Camera could not be opened after retries.\033[0m")
    return None