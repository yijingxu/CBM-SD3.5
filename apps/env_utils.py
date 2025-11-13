# env_utils.py
from dotenv import load_dotenv
import os

def get_env_var(key: str):
    load_dotenv()
    return os.getenv(key)
