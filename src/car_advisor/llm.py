import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

MODEL = os.environ.get("CAR_ADVISOR_MODEL", "anthropic:claude-sonnet-4-5")

_model = None


def model():
    global _model
    if _model is None:
        _model = init_chat_model(MODEL, max_tokens=2000)
    return _model
