from src.smtp_send import send_csv_email
from src.process_frames import get_data
from utils import open_camera, get_time
from multiprocessing import Process
from dotenv import load_dotenv
import schedule
import time
import cv2
import os

load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "CSV Report")
EMAIL_BODY = os.getenv("EMAIL_BODY", "Attached is the CSV report.")
CAM_URLS = {
    "ENTRANCE": os.getenv("ENTRANCE_CAM"),
    "EXIT": os.getenv("EXIT_CAM"),
}
MAX_FAILS = 5  



def job():
    today_date = get_time(date=True)
    try:
        send_csv_email(
            sender_email=SENDER_EMAIL,
            app_password=APP_PASSWORD,
            receiver_email=RECEIVER_EMAIL,
            subject=EMAIL_SUBJECT,
            body=EMAIL_BODY,
            csv_file_path=f"logs/{today_date}_log.csv"
        )
        print(f"[INFO] Email sent for {today_date}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")


def run_scheduler():
    schedule.every().day.at("19:30").do(job)
    while True:
        schedule.run_pending()
        time.sleep(60)



def run_camera(cam_name):
    cam = CAM_URLS.get(cam_name.upper())
    if cam is None:
        exit(f"[CRITICAL] Invalid camera name: {cam_name}")

    cap = open_camera(cam)
    if cap is None:
        exit("[CRITICAL] Unable to start camera feed. Exiting...")

    frame_count = 0
    fail_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"\033[93m[WARN] [{cam_name}] Failed to grab frame.\033[0m")
                time.sleep(1.5)
                fail_count += 1

                if fail_count >= MAX_FAILS:
                    print(f"\033[91m[CRITICAL] [{cam_name}] Reinitializing after {fail_count} fails...\033[0m")
                    cap.release()
                    cap = open_camera(cam)
                    fail_count = 0
                continue

            fail_count = 0
            frame_count += 1

            get_data(frame, cam_name)

    except KeyboardInterrupt:
        print(f"\n[INFO] [{cam_name}] Interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\n[ERROR] [{cam_name}] Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] [{cam_name}] Camera released and windows destroyed.")


if __name__ == "__main__":
    
    p1 = Process(target=run_camera, args=("Entrance",))
    p2 = Process(target=run_camera, args=("Exit",))

    
    p3 = Process(target=run_scheduler)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
