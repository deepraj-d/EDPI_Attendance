from utils import get_employee_name_arcface,get_time 
from src.embeddings import db_path,load_db
from src.embeddings import pad_crop
from ultralytics import YOLO
import logging
import os
import cv2
import csv
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

yolo_model_face = YOLO('pre_trained_models/yolov11n-face.pt')
yolo_model_body = YOLO('pre_trained_models/yolov8n.pt')

from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', allowed_modules=['recognition','detection'],providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

date_t = get_time(date=True)


current_csv_date = None
csv_file_path = None

def log_entries(new_entries):
    global current_csv_date, csv_file_path
    today_str = get_time(date=True)

    # If date changes, create a new file
    if today_str != current_csv_date:
        current_csv_date = today_str
        csv_file_path = f"logs/{today_str}_log.csv"
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Date','Name','Time',"Cam"])

    # Append the new entries
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_entries)





# ---- Embeddings ----
known_faces = load_db(db_path)


def get_data(frame,cam):
    """
    get data fuction is responsible for getting face detection
    two step detction step one will detect person and step two will  detect face
    
    """
    if cam=="Entrance":
        cv2.imwrite('entrance.jpg',frame)
    elif cam == "Exit":
        cv2.imwrite('exit.jpg',frame)
    new_entries = []
    name = None
    results = yolo_model_body(frame,verbose = False)
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            if cls_id != 0:  # 0 is 'person' in COCO
                continue

            conf = box.conf.item()
            if conf < 0.8:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_crop = frame[y1:y2, x1:x2]
            if person_crop is None:
                continue

            # face_found = False
            x1_f = y1_f = x2_f = y2_f = 0

            # ------- face detection --------
            if person_crop.size > 0:
                
                
                results2 = yolo_model_face(person_crop,verbose = False)
                for res in results2:
                    boxes_face = res.boxes
                    
                    for i,b in enumerate(boxes_face):
                        
                        conf_face = b.conf.item()
                        if conf_face < 0.8:
                            continue

                        # Face coordinates relative to the person_crop
                        fx1, fy1, fx2, fy2 = map(int, b.xyxy[0].tolist())

                        # Convert to original frame coordinates
                        x1_f, y1_f = x1 + fx1, y1 + fy1
                        x2_f, y2_f = x1 + fx2, y1 + fy2

                        
                        padded_face = pad_crop(frame, x1_f, y1_f, x2_f, y2_f, padding=20)

                        name,score = get_employee_name_arcface(padded_face,known_faces=known_faces,model_app=app)
                        
                        # if name is not None and score is not None:
                        if name is not None and score is not None:
                            name = name.split("_")[0] if "_" in name else name
                            new_entries.append([get_time(date=True),
                                                name,
                                                get_time(time=True),
                                                cam
                                                ])#, get_timestamp(fr=frame)])
                            logging.info(f"\033[92mPerson Identified: {name} at {get_time(time=True)}\033[0m")
                        # cv2.putText(frame, name, (x2_f, y2_f + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            # save_img_to_cluster(name,person_crop,name,frame_count)
                            # print(f"\033[91mPerson Identified {name} saved to cluster\033[0m")
    # Save new data to CSV
    if new_entries:
        log_entries(new_entries)

    
    return name





            

