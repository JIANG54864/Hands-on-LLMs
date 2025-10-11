import json
import re
from typing import Dict, List, Callable
import datetime


class BaseAgent:
    def __init__(self, llm_api: Callable, tools: Dict[str, Callable], memory_size: int = 5):
        """
        基础智能体类

        参数:
        llm_api: 大模型API调用函数
        tools: 可调用工具字典 {工具名: 工具函数}
        memory_size: 记忆保留的对话轮数
        """
        self.llm_api = llm_api
        self.tools = tools
        self.memory = []
        self.memory_size = memory_size

    def add_to_memory(self, role: str, content: str):
        """添加对话到记忆系统"""
        self.memory.append({"role": role, "content": content})
        # 保持记忆不超过指定大小
        if len(self.memory) > self.memory_size * 2:  # 每轮对话包含用户和AI两条
            self.memory = self.memory[-self.memory_size * 2:]

    def generate_prompt(self, user_input: str) -> List[Dict]:
        """生成包含记忆和当前输入的完整提示"""
        # 获取当前时间
        current_time = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
        
        # 在用户输入中添加当前时间信息
        timed_user_input = f"[当前时间: {current_time}] {user_input}"
        
        prompt = self.memory.copy()
        prompt.append({"role": "user", "content": timed_user_input})
        return prompt

    def parse_response(self, response: str) -> Dict:
        try:
            # 尝试解析JSON格式
            if response.strip().startswith("{") and response.strip().endswith("}") and "action" in response:
                parsed = json.loads(response)
                if parsed.get("action") in self.tools or parsed.get("action") == "reply":
                    return parsed
                else:
                    raise ValueError(f"Invalid action: {parsed.get('action')}")

            # 尝试提取内部嵌套的 Action 格式（如出现在 action_input 中）
            inner_action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
            inner_input_match = re.search(r"Input:\s*([^\n]+)", response, re.IGNORECASE)

            if inner_action_match:
                action = inner_action_match.group(1).strip()
                action_input = inner_input_match.group(1).strip() if inner_input_match else ""
                if action in self.tools:
                    return {
                        "thoughts": "从嵌套响应中提取到有效工具调用",
                        "action": action,
                        "action_input": action_input
                    }

            # 尝试解析非标准格式（原始级别匹配）
            thought_match = re.search(r"Thoughts?:\s*([^\n]*)", response, re.IGNORECASE)
            action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
            input_match = re.search(r"Input:\s*([^\n]+)", response, re.IGNORECASE)

            if action_match:
                action = action_match.group(1).strip()
                if action not in self.tools and action != "reply":
                    raise ValueError(f"Invalid action: {action}")

                return {
                    "thoughts": thought_match.group(1).strip() if thought_match else "",
                    "action": action,
                    "action_input": input_match.group(1).strip() if input_match else ""
                }

        except Exception as e:
            print(f"[Parse Error] {e}")
            pass

        # 默认返回直接回复
        return {
            "thoughts": "",
            "action": "reply",
            "action_input": response
        }


    def run_tool(self, tool_name: str, tool_input: str) -> str:
        """执行指定工具"""
        if tool_name not in self.tools:
            return f"Error: 未知工具 '{tool_name}'"

        try:
            # 尝试解析JSON格式输入
            try:
                params = json.loads(tool_input)
                if isinstance(params, dict):
                    return self.tools[tool_name](**params)
            except:
                pass

            # 尝试作为字符串参数调用
            # 如果输入为空，则不传递参数（让默认参数生效）
            if tool_input.strip() == "":
                return self.tools[tool_name]()
            else:
                return self.tools[tool_name](tool_input)
        except Exception as e:
            return f"Error: 工具执行失败 - {str(e)}"

    def process_tool_calls(self, prompt: List[Dict]) -> str:
        """处理工具调用，支持连续调用"""
        # 调用LLM获取响应
        llm_response = self.llm_api(prompt)
        
        # 解析LLM响应
        parsed = self.parse_response(llm_response)
        print(f"[DEBUG] Parsed response: {parsed}")

        # 处理工具调用
        if parsed["action"] != "reply":
            if parsed["action"] in self.tools:
                # 执行工具
                tool_result = self.run_tool(parsed["action"], parsed["action_input"])
                print(f"[DEBUG] Tool result: {tool_result}")

                # 将工具结果添加到记忆
                self.add_to_memory("assistant", f"工具调用结果: {tool_result}")

                # 创建新提示让LLM处理工具结果
                new_prompt = prompt.copy()
                new_prompt.append({"role": "assistant", "content": llm_response})
                new_prompt.append({"role": "user", "content": f"工具调用结果: {tool_result}"})

                # 递归处理可能的进一步工具调用
                return self.process_tool_calls(new_prompt)
            else:
                print(f"[ERROR] 工具 '{parsed['action']}' 不存在或未注册")
                return f"错误：工具 '{parsed['action']}' 未找到，请确认是否已注册该工具"
        else:
            # 直接回复
            self.add_to_memory("assistant", parsed["action_input"])
            return parsed["action_input"]

    def process(self, user_input: str) -> str:
        """处理用户输入并返回响应"""
        # 1. 生成完整提示
        prompt = self.generate_prompt(user_input)
        print(f"[DEBUG] Prompt sent to LLM: {prompt}")

        # 2. 处理工具调用（支持连续调用）
        return self.process_tool_calls(prompt)