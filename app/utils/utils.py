from passlib.context import CryptContext
import uuid

from ai_core.llm_api_provider import LlmApiProvider

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def generate_conversation_id():
    return uuid.uuid4() 


def is_empty(value):
    if isinstance(value, (str, list, tuple, dict, set)):
        # Check if the collection type (like string, list, dict) is empty
        return len(value) == 0
    # elif isinstance(value, int):
    #     # For integers, check if they are zero
    #     return value == 0
    elif value is None:
        # Handle None
        return True
    else:
        # If none of the above types, assume the value is not empty
        return False
