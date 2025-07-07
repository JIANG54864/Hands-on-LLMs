from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="qwen3:0.6b",
    messages=[
        {"role": "user", "content": "9.8和9.11哪个大"},
    ],
    max_tokens=1000,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    },
)
DEBUG = False  # 控制是否输出完整响应

if DEBUG:
    print("Full chat response:", chat_response)
else:
    print("Chat response:", chat_response.choices[0].message.content)
