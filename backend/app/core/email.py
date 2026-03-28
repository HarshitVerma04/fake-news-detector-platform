"""
backend/app/core/email.py
"""

import logging
from backend.app.core.config import settings

log = logging.getLogger(__name__)


async def send_password_reset_email(email: str, username: str, token: str):
    if not settings.MAIL_USERNAME or not settings.MAIL_FROM:
        log.warning(
            f"Email not configured. Password reset token for {email}: {token}\n"
            "To enable email, set MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM in .env"
        )
        return

    from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType

    conf = ConnectionConfig(
        MAIL_USERNAME=settings.MAIL_USERNAME,
        MAIL_PASSWORD=settings.MAIL_PASSWORD,
        MAIL_FROM=settings.MAIL_FROM,
        MAIL_PORT=settings.MAIL_PORT,
        MAIL_SERVER=settings.MAIL_SERVER,
        MAIL_FROM_NAME=settings.MAIL_FROM_NAME,
        MAIL_STARTTLS=settings.MAIL_STARTTLS,
        MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True,
    )

    reset_url = f"{settings.BASE_URL}/reset-password?token={token}"

    html = f"""
    <div style="font-family: monospace; max-width: 480px; margin: 40px auto; color: #333;">
        <h2 style="color: #d4a847;">SatyaParichay</h2>
        <p>Hello <strong>{username}</strong>,</p>
        <p>A password reset was requested for your account.</p>
        <p style="margin: 24px 0;">
            <a href="{reset_url}"
               style="background: #d4a847; color: #000; padding: 12px 24px;
                      text-decoration: none; border-radius: 4px; font-weight: bold;">
                Reset Password
            </a>
        </p>
        <p style="color: #999; font-size: 12px;">
            This link expires in 1 hour. If you did not request this, ignore this email.
        </p>
    </div>
    """

    message = MessageSchema(
        subject="Reset your SatyaParichay password",
        recipients=[email],
        body=html,
        subtype=MessageType.html,
    )

    try:
        await FastMail(conf).send_message(message)
        log.info(f"Password reset email sent to {email}")
    except Exception as e:
        log.error(f"Failed to send reset email to {email}: {e}")
        raise