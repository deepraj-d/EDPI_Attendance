import time
import cv2
from src.process_frames import get_data


# ---- Setup RTSP camera ----
# ffplay -rtsp_transport tcp 'rtsp://admin:admin123@192.168.29.247:554/rtsp/streaming?channel=07&subtype=A2' # for exit cam
# Stream #0:0: Video: hevc (Main), yuv420p(tv), 2688x1520, 25 fps, 25 tbr, 90k tbn
camera_url_entrance = 'rtsp://admin:admin@192.168.29.250:554/rtsp/streaming?channel=01&subtype=A2'
camera_url_exit = 'rtsp://admin:admin123@192.168.29.247:554/rtsp/streaming?channel=07&subtype=A2'

file_name = "Live_Stream"
MAX_FAILS = 5  # consecutive failed reads before restart


def open_camera(url, retries=3, delay=2):
    for attempt in range(retries):
        cap = cv2.VideoCapture(url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"\033[94m[INFO] Camera FPS: {fps:.2f}\033[0m")
        if cap.isOpened():
            print("\033[92m[INFO] Camera connected.\033[0m")
            return cap
        print(f"\033[93m[WARN] Retry {attempt + 1}/{retries} - Could not open stream. Retrying in {delay}s...\033[0m")
        time.sleep(delay)
    print("\033[91m[ERROR] Camera could not be opened after retries.\033[0m")
    return None


def main():
    cap = open_camera(camera_url_entrance)
    if cap is None:
        exit("[CRITICAL] Unable to start camera feed. Exiting...")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)

    frame_count = 0
    fail_count = 0

    while True:
        if cap is None:
            print("\033[91m[ERROR] Capture object is None. Attempting to reconnect...\033[0m")
            cap = open_camera(camera_url_entrance)
            continue

        ret, frame = cap.read()

        if not ret or frame is None:
            print("\033[93m[WARN] Failed to grab frame.\033[0m")
            time.sleep(1.5)
            ret, frame = cap.read()

            if not ret or frame is None:
                print("\033[91m[ERROR] Still unable to grab frame. Skipping...\033[0m")
                fail_count += 1

                if fail_count >= MAX_FAILS:
                    print(f"\033[91m[CRITICAL] Reinitializing capture after {fail_count} failed attempts...\033[0m")
                    cap.release()
                    cap = open_camera(camera_url_entrance)
                    fail_count = 0
                continue

            print("\033[92m[FIX] Recovered. Capturing frames again.\033[0m")

        # Reset fail count on successful frame
        fail_count = 0
        frame_count += 1
        # if frame_count % 5 == 0:
        #     get_data(frame, frame_count)
        get_data(frame, frame_count)

        # Optional: display the frame
        # cv2.imshow("Office_Entrance_Cam", frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
