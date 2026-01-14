from openai import OpenAI
import os

proxy = "http://127.0.0.1:7890"

# 如果需要代理（Windows），设置环境变量
if proxy:
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    
api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
  api_key=api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

models = client.models.list()
for model in models:
  print(model.id)