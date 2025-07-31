
import face_recognition
from datetime import datetime
import numpy as np
import cv2
import os 

# def get_embeddings_foo(s): 
#     name = s.split("/")[-1].split(".")[0]
#     image = cv2.imread(s)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_image)
#     face_encodings = face_recognition.face_encodings(rgb_image)
    
#     known_faces[name] = face_encodings[0]
#     return None

def get_employee_name(student_face,known_faces):
    """
    Detects faces in `person_crop` using YOLO `model`, matches embeddings with `known_faces`,
    and returns recognized name, similarity score, and face bbox in full-frame coordinates.
    
    `offset_x` and `offset_y` are used to convert crop-relative bbox to original frame coords.
    """
   
    # Crop face from original frame with padding
    
    resized_face = cv2.resize(student_face, (150, 150))

    embedding = get_embedding(resized_face)
    recognized_name = None
    similarity_sc = None

    if embedding is not None:
        for name, known_embedding in known_faces.items():
            similarity = cosine_similarity(embedding, known_embedding)
            # print(f"Comparing with {name}: {similarity:.4f}")
            if similarity > 0.94:
                similarity_sc = similarity
                recognized_name = name
                break  # Optional: exit loop on first match

    if recognized_name is not None and similarity_sc is not None:
        return recognized_name, similarity_sc

    # No face detected or no match
    return None, None

import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity



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



# # ---- Helper: compute cosine similarity ----
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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


# ---- Helper: get 128D embedding ----
def get_embedding(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)#,number_of_times_to_upsample=2)
    encodings = face_recognition.face_encodings(rgb_img)#,face_locations,num_jitters=10,model='large')
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None
    
def get_time():
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    return time_str


# ---- Helper: temp time obj ----
import easyocr
reader = easyocr.Reader(['en'])
time_stamp_region = [1260,1,1420,40]
def get_timestamp(fr):
    region = fr[time_stamp_region[1]:time_stamp_region[3],
                   time_stamp_region[0]:time_stamp_region[2]]
    out = reader.readtext(image=region)
    time = out[0][1].replace(".",":")
    return time