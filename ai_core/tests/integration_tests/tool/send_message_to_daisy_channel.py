from langchain_core.tools import tool
import requests
import json


@tool
def send_message_to_daisy_channel(message: str, thread_ts: str = "") -> str:
    """
    Send a message to a Daisy slack channel. If the message_id is provided, the message will be replied to the message.

    Args:
        message (str): The message to send to the Slack channel
        thread_ts (str, optional): The ID of the message to reply to. Defaults to empty string

    Returns:
        str: A response message containing the message ID if successful, or error message if failed
    """
    url = "https://slack.com/api/chat.postMessage"
    token = "xoxb-1243672316950-6575404681559-zRPmoNQ7hzGIhx1wsw8ZlQex"
    channel = "C07D6H940P6"
    # channel = "C07AMEQEMJL"
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    payload = {
        "channel": channel,
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]
    }


    # url = "https://hooks.slack.com/services/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"
    #
    # proxies = {
    #     'http': None,
    #     'https': None
    # }
    #
    # payload = {
    #     "text": message
    # }
    #
    # if message_id:
    #     payload["thread_ts"] = message_id
    #
    response = requests.post(
        url,
        data=json.dumps(payload),
        headers=headers
        # proxies=proxies
    )

    if response.status_code == 200:
        response_data = response.json()
        message_timestamp = response_data.get('ts')
        return f"메시지가 성공적으로 전송되었습니다. 메시지 ID: {message_timestamp}"
    else:
        return f"메시지 전송 실패. 상태 코드: {response.status_code}, 응답: {response.text}"

