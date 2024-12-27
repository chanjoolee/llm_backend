from langchain_core.tools import tool
import requests
import json


@tool
def send_message_to_daisy_channel(message: str, message_id: str = "") -> str:
    """
    Send a message to a Daisy slack channel. If the message_id is provided, the message will be replied to the message.

    Args:
        message (str): The message to send to the Slack channel
        message_id (str, optional): The ID of the message to reply to. Defaults to empty string

    Returns:
        str: A response message containing the message ID if successful, or error message if failed
    """

    def _send_message_to_daisy_channel(message: str, message_id: str = "") -> str:
        url = "https://slack.com/api/chat.postMessage"
        token = "xoxb-1243672316950-6575404681559-zRPmoNQ7hzGIhx1wsw8ZlQex"

        payload = {
            "channel": "C07D6H940P6",
            "text": message
        }

        if message_id:
            payload["thread_ts"] = message_id

        try:
            response = requests.post(
                url,
                data=json.dumps(payload),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {token}'
                },
            )

            if response.status_code == 200:
                response_data = response.json()
                message_timestamp = response_data.get('ts')
                if not response_data['ok']:
                    return f"메시지 전송 실패. 응답: {response.text}"

                return f"메시지가 성공적으로 전송되었습니다. 메시지 ID: {message_timestamp}"
            else:
                return f"메시지 전송 실패. 상태 코드: {response.status_code}, 응답: {response.text}"

        except Exception as e:
            return f"메시지 전송 중 오류가 발생했습니다: {str(e)}"


    # if message length is greater than 1000, split the message into multiple messages
    if len(message) > 1000:
        message_parts = [message[i:i + 1000] for i in range(0, len(message), 1000)]
        for part in message_parts:
            _send_message_to_daisy_channel(part, message_id)

        return "메시지가 성공적으로 전송되었습니다."

    return _send_message_to_daisy_channel(message, message_id)