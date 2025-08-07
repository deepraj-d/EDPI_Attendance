from email.message import EmailMessage
from dotenv import load_dotenv
from datetime import datetime
import smtplib
import os

# SEND_INTERVAL = int(os.getenv("SEND_INTERVAL", 300))  # default to 5 minutes
def get_time(date=False, time=False):
    now = datetime.now()

    if date and time:
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif date:
        return now.strftime("%Y-%m-%d")
    elif time:
        return now.strftime("%H:%M:%S")
    else:
        return "Please specify either date or time"
    
 
def send_csv_email(
    sender_email: str,
    app_password: str,
    receiver_email: str,
    subject: str,
    body: str,
    csv_file_path: str
):
    # Ensure file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    # Create email message
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    # Read and attach the CSV file
    with open(csv_file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(csv_file_path)
        msg.add_attachment(
            file_data,
            maintype="application",
            subtype="octet-stream",
            filename=file_name
        )

    # Send the email using Gmail's SMTP server
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)

    print(f"Email sent to {receiver_email} with attachment: {file_name}")



