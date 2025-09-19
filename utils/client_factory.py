import os
from google import genai

def create_gemini_client(api_key: str, proxy: str = None):
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
    return genai.Client(api_key=api_key)
