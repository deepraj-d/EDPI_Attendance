import os
import cv2
import json
import face_recognition

db = {}

# ---- Helper: get 128D embedding ----
def get_embedding(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img,number_of_times_to_upsample=2)  # 2 tested
    encodings = face_recognition.face_encodings(rgb_img,face_locations,num_jitters=10,model='large')  # 10 tested
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None


folder_name = 'refrence_images'

for i in os.listdir(folder_name):
    if i.startswith("."):
        continue
    
    path = os.path.join(folder_name,i)
    image = cv2.imread(path)
    name = i.split(".")[0]
    
    embeddings = get_embedding(image)
    print(f"file {i} done")
    if embeddings.size > 0:
        db[name] = embeddings.tolist()
    else:
        print(f"\033[31mRed {i} not embedded\033[0m")      

        continue

with open("db/embeddings_db.json", "w") as f:
    json.dump(db, f)



    
