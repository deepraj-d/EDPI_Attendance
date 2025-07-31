from src.process_frames import get_data
import cv2



# ---- Setup video input/output ----
camera_url = 'rtsp://admin:admin@192.168.29.250:554/rtsp/streaming?channel=01&subtype=A2'
file_name="Live_Stream"


def main():
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("Cap not Found")
        exit()
            
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)


    frame_count = 0
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        get_data(frame, frame_count)

        cv2.imshow("Office_Entrance_Cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()