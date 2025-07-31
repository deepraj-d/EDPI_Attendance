from src.process_frames import get_data
import cv2
import os



# ---- Setup video input/output ----
path_video = 'Input_data/Day4.mp4'
file_name = os.path.splitext(os.path.basename(path_video))[0]


cap = cv2.VideoCapture(path_video)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)
writer = cv2.VideoWriter(f"output_data/{file_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, size)



# ---- door - detection ----
frame_count = 0
door_open = False  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    get_data(frame, frame_count)

    cv2.rectangle(frame, (1260,1), (1420,40), (255, 0, 0), 2)
    cv2.imshow("Office_Entrance_Cam", frame)

    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
