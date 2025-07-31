
import face_recognition
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




# ---- Helper: compute cosine similarity ----
def cosine_similarity(a, b):
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
    face_locations = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img,face_locations)
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None
    
# ---- Saving Full Body Image ----
def save_img_to_cluster(cluster_name,body_image,name,frame_number):
    
    os.makedirs(f"Body_images/{cluster_name}",exist_ok=True)

    cv2.imwrite(f'Body_images/{name}/{name}_{frame_number}.jpg',body_image)

    return None