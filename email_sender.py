"""
email_sender.py
───────────────
Send emails via SMTP with PDF attachment and log results to CSV.
"""

import csv
import os
import smtplib
import time
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

LOG_FILE = "sent_emails_log.csv"
LOG_HEADERS = [
    "timestamp",
    "company",
    "recipient_email",
    "subject",
    "status",
    "error",
]


def send_email(
    to_email: str,
    subject: str,
    body: str,
    attachment_path: str | None,
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
) -> dict:
    """
    Send a single email with optional PDF attachment.

    Returns {"success": bool, "error": str | None}
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject

        # Convert body to HTML (supports bold via <b> tags)
        # If body already contains HTML tags, use as-is; otherwise wrap newlines
        if "<b>" in body or "<br>" in body:
            html_body = body.replace("\n", "<br>")
        else:
            html_body = body.replace("\n", "<br>")
        html_body = f"<div style='font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;'>{html_body}</div>"
        msg.attach(MIMEText(html_body, "html"))

        # Attach PDF if provided
        if attachment_path and os.path.isfile(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(attachment_path)}",
            )
            msg.attach(part)

        # Connect and send
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())

        return {"success": True, "error": None}

    except Exception as e:
        return {"success": False, "error": str(e)}


def log_sent_email(
    company: str,
    recipient_email: str,
    subject: str,
    success: bool,
    error: str | None = None,
    log_dir: str = ".",
) -> None:
    """Append a row to the CSV log file."""
    log_path = os.path.join(log_dir, LOG_FILE)
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "company": company,
                "recipient_email": recipient_email,
                "subject": subject,
                "status": "SENT" if success else "FAILED",
                "error": error or "",
            }
        )


def is_already_sent(
    recipient_email: str,
    log_dir: str = ".",
) -> bool:
    """Check the CSV log to see if we already sent to this email."""
    log_path = os.path.join(log_dir, LOG_FILE)
    if not os.path.isfile(log_path):
        return False
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("recipient_email") == recipient_email
                and row.get("status") == "SENT"
            ):
                return True
    return False


def send_reply(
    to_email: str,
    subject: str,
    body: str,
    message_id: str,
    in_reply_to: str,
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    cc: str = "",
    bcc: str = "",
    attachment_paths: list[str] | None = None,
) -> dict:
    """
    Send a reply email that preserves the thread.

    Uses In-Reply-To and References headers so the reply appears
    in the same conversation thread in Gmail / Outlook / etc.

    Returns {"success": bool, "error": str | None}
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        if cc:
            msg["Cc"] = cc
        # Subject — prepend Re: if not already present
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        # Thread headers
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
        if message_id:
            msg["References"] = message_id

        # Body
        if "<b>" in body or "<br>" in body or "<p>" in body:
            html_body = body.replace("\n", "<br>")
        else:
            html_body = body.replace("\n", "<br>")
        html_body = (
            f"<div style='font-family: Arial, sans-serif; "
            f"font-size: 14px; line-height: 1.6;'>{html_body}</div>"
        )
        msg.attach(MIMEText(html_body, "html"))

        # Attachments
        if attachment_paths:
            for path in attachment_paths:
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(path)}",
                    )
                    msg.attach(part)

        # Collect all recipients
        recipients = [to_email]
        if cc:
            recipients += [a.strip() for a in cc.split(",") if a.strip()]
        if bcc:
            recipients += [a.strip() for a in bcc.split(",") if a.strip()]

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())

        return {"success": True, "error": None}

    except Exception as e:
        return {"success": False, "error": str(e)}


def send_composed_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    cc: str = "",
    bcc: str = "",
    attachment_paths: list[str] | None = None,
) -> dict:
    """
    Send a freshly composed email with optional CC, BCC, and attachments.

    Returns {"success": bool, "error": str | None}
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        if cc:
            msg["Cc"] = cc
        msg["Subject"] = subject

        # Body
        if "<b>" in body or "<br>" in body or "<p>" in body:
            html_body = body.replace("\n", "<br>")
        else:
            html_body = body.replace("\n", "<br>")
        html_body = (
            f"<div style='font-family: Arial, sans-serif; "
            f"font-size: 14px; line-height: 1.6;'>{html_body}</div>"
        )
        msg.attach(MIMEText(html_body, "html"))

        # Attachments
        if attachment_paths:
            for path in attachment_paths:
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(path)}",
                    )
                    msg.attach(part)

        recipients = [to_email]
        if cc:
            recipients += [a.strip() for a in cc.split(",") if a.strip()]
        if bcc:
            recipients += [a.strip() for a in bcc.split(",") if a.strip()]

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipients, msg.as_string())

        return {"success": True, "error": None}

    except Exception as e:
        return {"success": False, "error": str(e)}


def rate_limited_sleep(seconds: float = 2.0) -> None:
    """Simple delay between consecutive sends to avoid throttling."""
    time.sleep(seconds)
