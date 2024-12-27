from passlib.context import CryptContext
import uuid

from ai_core.llm_api_provider import LlmApiProvider
from sqlalchemy.inspection import inspect
from pydantic import BaseModel
import random
import string

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
    
def sqlalchemy_to_dict(instance):
    return {c.key: getattr(instance, c.key) for c in inspect(instance).mapper.column_attrs}


def generate_random_password(length=12):
    """
    Generate a random password with the specified length.
    The password contains at least one uppercase letter, one digit, and one special character.
    """
    if length < 4:
        raise ValueError("Password length must be at least 4 characters")

    # Define character pools
    uppercase = random.choice(string.ascii_uppercase)  # Ensure at least one uppercase
    digit = random.choice(string.digits)              # Ensure at least one digit
    special = random.choice("!@#$%^&*()-_=+[]{}|;:,.<>?/")  # Ensure at least one special character
    remaining = random.choices(string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:,.<>?/", k=length-3)
    
    # Combine and shuffle to create a random password
    password = list(uppercase + digit + special + "".join(remaining))
    random.shuffle(password)
    return "".join(password)