from ultralytics import YOLO


door_detection_model = YOLO('pre_trained_models/door_open_model.pt')

def Qdoor(fr):
    res = door_detection_model(fr)
    for r in res:
        door_open = True if r[0].boxes.cls==1 else False
    return door_open