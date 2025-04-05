import json
from fastapi import HTTPException, APIRouter
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app import database
from app.model import model_llm
from app.schema import schema_llm
import app.config
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()

SMTP_SERVER = os.getenv('SMTP_SERVER')  # Change to production if needed
SMTP_PORT = int(os.getenv('SMTP_PORT'))
SMTP_SERVICEID = os.getenv('SMTP_SERVICEID')

@router.post(
    "/", 
    tags=["Session"],
    description="""
    비밀번호 초기화 이메일 테스트용
    
    seungbum.lee@sk.com
    jeongho.jang@sk.com
    kunwoo.kim@bluedigm.com
    kunwoo.kim@partner.sktelecom.com
    no-reply.jira@sktelecom.com
    
    내부발송
        sk.com메일도메인은SKT 내부구성원메일발송시에만사용하며, 
        수신자메일주소가외부메일일경우발송불가
        ex) mgs@sk.com(발신자메일주소) -> xxxxxxxxx@sk.com(수신자메일주소)
    외부발송
        메일도메인을sktelecom.com으로지정후메일발송
        ex) mgs@sktelecom.com(발신자메일주소) -> xxxxxx@naver.com(수신자메일주소)
    """
)
async def send_email(email_request: schema_llm.EmailRequest):
    
    logger.info(f"start send email : {email_request}" )
    # Generate a unique message ID
    unique_number = int(time.time() * 1000)  # Use timestamp to ensure uniqueness
    message_id = f"{SMTP_SERVICEID}_MGS_{unique_number}"
    # Establish SMTP connection
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()

    # Compose email
    msg = MIMEMultipart()
    msg["From"] = email_request.sender_email
    msg["To"] = email_request.receiver_email
    msg["Subject"] = email_request.subject
    msg["Message-ID"] = message_id

    body = MIMEText(email_request.content, "plain", "utf-8")
    msg.attach(body)

    # Send the email
    my_response = server.sendmail(
        from_addr=email_request.sender_email, 
        to_addrs=email_request.receiver_email, 
        msg=msg.as_string()
    )
    server.quit()

    return {
        "message": f"Email sent successfully {email_request}",
        "response" : json.dumps(my_response, ensure_ascii=False)
    }