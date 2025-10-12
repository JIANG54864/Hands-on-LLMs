import requests

from typing import Dict, List, Optional
import sys
import os
import datetime
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ToolLibrary import ToolLibrary
from BaseAgent import BaseAgent
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 现在可以通过 os.getenv 获取环境变量


class DeepSeekAPI:
    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError("DeepSeek API key not provided")

    def __call__(self, messages: List[Dict]) -> str:
        """调用DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
            "stop": None
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"API调用错误: {str(e)}"


class QwenAPI:
    def __init__(self, model: str = "qwen3-max", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("Qwen API key not provided")

    def __call__(self, messages: List[Dict]) -> str:
        """调用Qwen API"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"enable_thinking": False}
            )
            return completion.choices[0].message.content
        except ImportError:
            return "API调用错误: 未安装openai库，请运行 'pip install openai'"
        except Exception as e:
            return f"API调用错误: {str(e)}"


class OllamaAPI:
    def __init__(self, model: str = "myqwen"):
        self.api_url = "http://localhost:11434/api/chat"
        self.model = model

    def __call__(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": True
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            return f"Ollama调用错误: {str(e)}"



def main():
    # 1. 初始化LLM API
    # llm_api = OllamaAPI(model="myqwen")
    # llm_api = DeepSeekAPI(model="deepseek-chat")
    llm_api = QwenAPI(model="qwen3-max")

    # 2. 创建工具库
    tools = {
        "calculator": ToolLibrary.calculator,
        "time": ToolLibrary.time,
        "web_search": ToolLibrary.web_search,
        "unit_converter": ToolLibrary.unit_converter,
        "read_file": ToolLibrary.read_file,
        "rename_file": ToolLibrary.rename_file,
        "list_files": ToolLibrary.list_files
    }

    # 3. 创建智能体实例
    agent = BaseAgent(llm_api, tools, memory_size=3)

    # 4. 系统提示（初始化记忆）
    # 获取当前日期和时间
    current_time = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
    
    system_prompt = (
        f"当前时间是 {current_time}。\n"
        "你是一个智能助手，可以调用工具解决用户问题。\n"
        "你可以使用的工具包括: \n"
        "1. calculator - 计算数学表达式，如: '3+5*2'\n"
        "2. time - 获取当前时间或日期\n"
        "3. web_search - 搜索网络信息\n"
        "4. unit_converter - 单位转换，格式: '10 km to mi'\n"
        "5. list_files - 列出目录中的文件和文件夹\n"
        "6. read_file - 读取文件内容\n"
        "7. rename_file - 重命名文件\n"
        "\n"
        "请严格按照以下格式进行响应:\n"
        "Thoughts: 你的思考过程\n"
        "Action: 工具名|reply\n"
        "Input: 工具参数|回复内容\n"
        "\n"
        "示例1 - 调用时间工具:\n"
        "Thoughts: 用户想知道当前时间，我需要调用time工具\n"
        "Action: time\n"
        "Input: \n"
        "\n"
        "示例2 - 直接回复:\n"
        "Thoughts: 用户打招呼，我可以直接回复\n"
        "Action: reply\n"
        "Input: 你好！有什么可以帮助你的吗？\n"
        "\n"
        "示例3 - 工具执行后返回结果:\n"
        "（系统输入）工具调用结果: 目录 '.' 中的文件和文件夹:文件夹:  文件夹: __pycache__/ 文件: main.py, utils.py\n"
        "Thoughts: 我已经得到了文件列表，现在可以告诉用户有哪些文件\n"
        "Action: reply\n"
        "Input: 当前文件夹中包含以下内容：文件夹: __pycache__/ 文件: main.py, utils.py\n"
        "\n"
        "示例4 - 计算器工具使用后:\n"
        "（系统输入）工具调用结果: 计算结果: 3+5*2 = 13\n"
        "Thoughts: 我已经得到了计算结果，现在可以告诉用户\n"
        "Action: reply\n"
        "Input: 计算结果是：3+5*2 = 13\n"
        "\n"
        "示例5 - 单位转换工具使用后:\n"
        "（系统输入）工具调用结果: 10 km = 6.21 mi\n"
        "Thoughts: 我已经得到了单位转换结果，现在可以告诉用户\n"
        "Action: reply\n"
        "Input: 单位转换结果是：10 km = 6.21 mi\n"
        "\n"
        "示例6 - 网络搜索工具使用后:\n"
        "（系统输入）工具调用结果: 搜索'最新科技新闻'的结果: 1. 标题: 人工智能新突破... 内容摘要: 近日...\n"
        "Thoughts: 我已经得到了搜索结果，现在可以告诉用户\n"
        "Action: reply\n"
        "Input: 这是我为您找到的关于最新科技新闻的信息：1. 标题: 人工智能新突破... 内容摘要: 近日...\n"
        "\n"
        "重要提醒：你必须严格按照上述格式进行响应，不要添加任何额外的内容。在回复用户时，必须包含工具返回的具体内容，不能遗漏关键信息。"
    )
    agent.add_to_memory("system", system_prompt)

    # 5. 交互循环
    print("智能体已启动。输入 'exit' 退出。")
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "exit":
            break

        # 处理用户输入
        response = agent.process(user_input)
        print(f"智能体: {response}")


if __name__ == "__main__":
    main()