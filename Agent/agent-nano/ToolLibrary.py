import math
import datetime
import requests
from urllib.parse import quote
import os
from bs4 import BeautifulSoup
import re

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
        # 无条件返回日期和时间
        return f"当前日期和时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"

    @staticmethod
    def web_search(query: str) -> str:
        """简单的网页搜索"""
        print(f"（调用网页搜索工具）")
        try:
            # 使用百度搜索获取结果
            search_url = f"https://www.baidu.com/s?wd={quote(query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # 解析搜索结果
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 定义需要过滤的无用网站列表
            filtered_domains = [
                'hao123.com', 'baidu.com', 'baike.com', 'zhidao.baidu.com', 
                'tieba.baidu.com', 'jingyan.baidu.com'
            ]
            
            # 提取前几个搜索结果链接
            links = []
            for link_elem in soup.find_all('a', href=True):
                href = link_elem['href']
                # 获取实际链接（百度搜索结果中的链接可能需要进一步处理）
                if 'link?url=' in href:
                    # 这是百度的跳转链接，需要访问以获取真实链接
                    try:
                        real_response = requests.head(href, allow_redirects=True, timeout=5)
                        real_url = real_response.url
                    except:
                        real_url = href
                else:
                    real_url = href
                    
                # 过滤出有效的外部链接
                if real_url.startswith('http') and not any(domain in real_url for domain in filtered_domains):
                    title = link_elem.get_text().strip()
                    if title and len(title) > 5:  # 过滤掉标题太短的链接
                        links.append({'url': real_url, 'title': title})
                        if len(links) >= 5:  # 获取更多链接以确保有足够的有效链接
                            break
            
            # 如果没有找到有效链接，尝试其他方法
            if not links:
                # 尝试查找h3标签下的链接
                for h3_elem in soup.find_all('h3'):
                    link_elem = h3_elem.find('a', href=True)
                    if link_elem:
                        href = link_elem['href']
                        # 处理百度跳转链接
                        if 'link?url=' in href:
                            try:
                                real_response = requests.head("https://www.baidu.com" + href, allow_redirects=True, timeout=5)
                                real_url = real_response.url
                            except:
                                real_url = href
                        else:
                            real_url = href
                            
                        title = link_elem.get_text().strip()
                        if (real_url.startswith('http') and 
                            not any(domain in real_url for domain in filtered_domains) and 
                            title and len(title) > 5):
                            links.append({'url': real_url, 'title': title})
                            if len(links) >= 5:
                                break
            
            # 访问这些链接并获取内容摘要
            results = []
            for link_info in links:
                try:
                    # 检查链接是否在过滤列表中
                    if any(domain in link_info['url'] for domain in filtered_domains):
                        continue
                        
                    page_response = requests.get(link_info['url'], headers=headers, timeout=8)
                    page_response.raise_for_status()
                    
                    page_soup = BeautifulSoup(page_response.text, 'html.parser')
                    
                    # 移除脚本和样式元素
                    for script in page_soup(["script", "style"]):
                        script.decompose()
                    
                    # 获取页面标题
                    page_title = ""
                    title_tag = page_soup.find('title')
                    if title_tag:
                        page_title = title_tag.get_text().strip()
                    
                    # 获取页面主要内容
                    page_content = ""
                    # 尝试多种方法获取主要内容
                    content_candidates = [
                        page_soup.find('article'),
                        page_soup.find('main'),
                        page_soup.find('div', class_='content'),
                        page_soup.find('div', class_='post-content'),
                        page_soup.find('div', class_='article-content'),
                        page_soup.find('div', {'id': 'content'}),
                        page_soup.find('div', {'class': 'post'}),
                        page_soup.find('body')
                    ]
                    
                    for candidate in content_candidates:
                        if candidate:
                            # 提取文本并清理
                            text = candidate.get_text()
                            # 移除多余的空白字符
                            text = re.sub(r'\s+', ' ', text).strip()
                            if len(text) > 100:  # 只有足够长的文本才算有效
                                # 取前300个字符作为摘要
                                page_content = text[:300] + ("..." if len(text) > 300 else "")
                                break
                    
                    if (page_title and len(page_title) > 10) or (page_content and len(page_content) > 50):
                        results.append({
                            'title': link_info['title'] if len(link_info['title']) > 10 else page_title,
                            'desc': page_content if page_content else "无内容摘要",
                            'link': link_info['url']
                        })
                        
                    # 如果已经获取到足够的结果，就停止
                    if len(results) >= 3:
                        break
                        
                except Exception as e:
                    # 如果访问某个链接失败，继续尝试下一个
                    continue
            
            # 格式化结果
            if results:
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. 标题: {result['title']}\n"
                        f"   内容摘要: {result['desc']}\n"
                        f"   链接: {result['link']}\n"
                    )
                return f"搜索'{query}'的结果:\n" + "\n".join(formatted_results)
            else:
                # 如果所有方法都失败了，至少返回搜索链接
                return f"已在百度搜索 '{query}'，请访问以下链接查看结果：{search_url}"
            
        except requests.exceptions.Timeout:
            return "搜索超时，请稍后重试"
        except requests.exceptions.RequestException as e:
            return f"网络请求错误: {str(e)}"
        except Exception as e:
            return f"搜索过程中发生错误: {str(e)}"

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

    @staticmethod
    def read_file(file_path: str) -> str:
        """读取文件内容"""
        print(f"（调用读取文件工具）")
        try:
            if not os.path.exists(file_path):
                return f"错误: 文件 '{file_path}' 不存在"

            if not os.path.isfile(file_path):
                return f"错误: '{file_path}' 不是一个有效文件"

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            return f"读取文件时发生错误: {str(e)}"

    @staticmethod
    def rename_file(old_path: str, new_path: str) -> str:
        """重命名文件"""
        print(f"（调用重命名文件工具）")
        try:
            if not os.path.exists(old_path):
                return f"错误: 原文件 '{old_path}' 不存在"

            if os.path.exists(new_path):
                return f"错误: 目标文件 '{new_path}' 已存在"

            os.rename(old_path, new_path)
            return f"成功将 '{old_path}' 重命名为 '{new_path}'"
        except Exception as e:
            return f"重命名文件时发生错误: {str(e)}"

    @staticmethod
    def list_files(directory: str = ".") -> str:
        """列出目录中的文件和文件夹"""
        print(f"（调用列出文件工具）")
        try:
            if not os.path.exists(directory):
                return f"错误: 目录 '{directory}' 不存在"

            if not os.path.isdir(directory):
                return f"错误: '{directory}' 不是一个有效目录"

            items = os.listdir(directory)
            if not items:
                return f"目录 '{directory}' 为空"

            # 分离文件和文件夹
            files = []
            folders = []
            for item in items:
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    files.append(item)
                else:
                    folders.append(item)

            result = f"目录 '{directory}' 中的文件和文件夹:\n"
            if folders:
                result += "文件夹:\n" + "\n".join([f"  {folder}/" for folder in folders]) + "\n"
            if files:
                result += "文件:\n" + "\n".join([f"  {file}" for file in files])

            return result
        except Exception as e:
            return f"列出文件时发生错误: {str(e)}"