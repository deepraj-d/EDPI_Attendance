import time
import cv2
import os
from multiprocessing import Process
from dotenv import load_dotenv
from src.process_frames import get_data
from utils import open_camera

load_dotenv()

CAM_URLS = {
    "ENTRANCE": os.getenv("ENTRANCE_CAM"),
    "EXIT": os.getenv("EXIT_CAM"),
}

MAX_FAILS = 5  # consecutive failed reads before restart


def run_camera(cam_name):
    cam = CAM_URLS.get(cam_name.upper())
    if cam is None:
        exit(f"[CRITICAL] Invalid camera name: {cam_name}")

    cap = open_camera(cam)
    if cap is None:
        exit("[CRITICAL] Unable to start camera feed. Exiting...")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)

    frame_count = 0
    fail_count = 0

    while True:
        if cap is None:
            print(f"\033[91m[ERROR] [{cam_name}] Capture object is None. Attempting to reconnect...\033[0m")
            cap = open_camera(cam)
            continue

        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"\033[93m[WARN] [{cam_name}] Failed to grab frame.\033[0m")
            time.sleep(1.5)
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"\033[91m[ERROR] [{cam_name}] Still unable to grab frame. Skipping...\033[0m")
                fail_count += 1

                if fail_count >= MAX_FAILS:
                    print(f"\033[91m[CRITICAL] [{cam_name}] Reinitializing capture after {fail_count} failed attempts...\033[0m")
                    cap.release()
                    cap = open_camera(cam)
                    fail_count = 0
                continue

            print(f"\033[92m[FIX] [{cam_name}] Recovered. Capturing frames again.\033[0m")

        fail_count = 0
        frame_count += 1

        
        get_data(frame, cam_name)

        # Optional: display the frame
        # cv2.imshow(cam_name, frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p1 = Process(target=run_camera, args=("Entrance",))
    p2 = Process(target=run_camera, args=("Exit",))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
