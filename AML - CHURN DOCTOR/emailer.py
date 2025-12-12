# emailer.py
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple

# Email configuration - can be set via environment variables or directly
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("app email", "app email")
SMTP_PASS = os.getenv("app password", "app password")

def send_email(
    to_email: str, 
    subject: str, 
    body: str,
    smtp_user: Optional[str] = None,
    smtp_pass: Optional[str] = None,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Send an email with the churn report.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body text
        smtp_user: SMTP username (optional, uses config if not provided)
        smtp_pass: SMTP password (optional, uses config if not provided)
        smtp_host: SMTP host (optional, uses config if not provided)
        smtp_port: SMTP port (optional, uses config if not provided)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    # Use provided values or fall back to config/environment
    user = smtp_user or SMTP_USER
    password = smtp_pass or SMTP_PASS
    host = smtp_host or SMTP_HOST
    port = smtp_port or SMTP_PORT
    
    # Check if email is configured
    if user == "YOUR_EMAIL@gmail.com" or password == "YOUR_APP_PASSWORD":
        return False, "Email not configured. Please set SMTP_USER and SMTP_PASS in emailer.py or environment variables."
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = user
        msg["To"] = to_email
        msg["Subject"] = subject
        
        # Add body
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        
        return True, f"Email sent successfully to {to_email}"
    
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Please check your email and app password."
    except smtplib.SMTPRecipientsRefused:
        return False, f"Invalid recipient email address: {to_email}"
    except smtplib.SMTPServerDisconnected:
        return False, "Connection to email server lost. Please check your internet connection."
    except Exception as e:
        return False, f"Error sending email: {str(e)}"


def is_email_configured() -> bool:
    """Check if email is properly configured."""
    return (SMTP_USER != "YOUR_EMAIL@gmail.com" and 
            SMTP_PASS != "YOUR_APP_PASSWORD" and
            bool(SMTP_USER) and bool(SMTP_PASS))
