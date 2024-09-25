# config.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()  # This will load from .env file by default
environment = os.getenv('ENVIRONMENT', 'local')

if environment == 'development':
    load_dotenv('.env.development')
elif environment == 'product':
    load_dotenv('.env.product')
else:
    load_dotenv('.env.local')
