import json

import requests

# Ollama 的基础 URL
OLLAMA_URL = "http://localhost:11434/api/chat"

# 请求参数
payload = {
    "model": "myqwen",
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ],
    "stream": True,
    "think": False
}

response = requests.post(OLLAMA_URL, json=payload)
if not payload["stream"]:
    result = response.json()
    print(result["message"]["content"])
else:
    for line in response.iter_lines():
        if line:
            try:
                result = json.loads(line)
                print(result["message"]["content"],end="",flush=True)
            except json.JSONDecodeError:
                print(line)
