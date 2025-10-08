import math
import datetime
import requests
from urllib.parse import quote
import os
import random

class ToolLibrary:
    @staticmethod
    def calculator(expression: str) -> str:
        """数学计算器"""
        print(f"（调用数学计算器）")
        try:
            # 安全计算 - 仅允许数学表达式
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "错误: 表达式包含非法字符"

            result = eval(expression, {"__builtins__": None}, math.__dict__)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    @staticmethod
    def time(query: str = "") -> str:
        """获取当前时间或日期"""
        print(f"（调用时间工具）")
        now = datetime.datetime.now()
        if "date" in query.lower() or "日期" in query.lower():
            return f"当前日期: {now.strftime('%Y-%m-%d')}"
        return f"当前时间: {now.strftime('%H:%M:%S')}"

    @staticmethod
    def web_search(query: str) -> str:
        """简单的网页搜索"""
        print(f"（调用网页搜索工具）")


    @staticmethod
    def unit_converter(query: str) -> str:
        """单位转换器"""
        print(f"（调用单位转换工具）")
        # 实现简单的单位转换逻辑
        conversions = {
            # 温度
            ("c", "f"): lambda c: c * 9 / 5 + 32,
            ("f", "c"): lambda f: (f - 32) * 5 / 9,
            # 长度
            ("km", "mi"): lambda km: km * 0.621371,
            ("mi", "km"): lambda mi: mi * 1.60934,
            ("m", "ft"): lambda m: m * 3.28084,
            ("ft", "m"): lambda ft: ft * 0.3048,
            # 重量
            ("kg", "lb"): lambda kg: kg * 2.20462,
            ("lb", "kg"): lambda lb: lb * 0.453592,
        }

        try:
            parts = query.lower().split()
            if len(parts) < 4:
                return "格式: [值] [原单位] to [目标单位]"

            value = float(parts[0])
            from_unit = parts[1]
            to_unit = parts[3]

            # 查找转换函数
            for (f, t), func in conversions.items():
                if from_unit == f and to_unit == t:
                    result = func(value)
                    return f"{value} {from_unit} = {result:.2f} {to_unit}"

            return f"不支持从 {from_unit} 到 {to_unit} 的转换"
        except Exception as e:
            return f"转换失败: {str(e)}"