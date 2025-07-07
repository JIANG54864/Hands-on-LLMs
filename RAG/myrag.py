from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def build_rag_prompt(user_query: str, context_chunks: list, max_context_length: int = 3000) -> str:
    # 步骤 1：合并上下文块
    context_str = "\n\n---\n\n".join(context_chunks)

    # 步骤 2：截断超长上下文（避免超出模型限制）
    if len(context_str) > max_context_length:
        context_str = context_str[:max_context_length] + "【...后续内容已截断】"

    # 步骤 3：应用模板
    return ADVANCED_PROMPT_TEMPLATE.format(
        context_str=context_str,
        user_query=user_query
    )

model = SentenceTransformer('shibing624/text2vec-base-chinese')

client = PersistentClient(path="./chroma")  # 确保路径正确
collection = client.get_collection(name="doupocangqiong1-500")

user_query = input("请输入您的问题: ")
query_embedding = model.encode([user_query])[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],  # 注意是 list of lists
    n_results=3  # 返回 top K 个结果
)
context_chunks = results['documents'][0]  # 取第一个查询的结果 (因为我们只查了一个)
retrieved_metadata = results['metadatas'][0]

# 模板，根据具体任务进行修改
ADVANCED_PROMPT_TEMPLATE = """
# 角色设定
你是一位资深小说读者，擅长根据文本证据进行严谨分析。
回答必须基于提供的上下文，严禁编造信息。

# 任务说明
1. 分析用户问题与上下文的关联性
2. 从上下文中提取相关证据
3. 组织语言清晰、简洁地回答

# 上下文信息
{context_str}

# 用户问题
{user_query}

# 回答要求
1. 开头直接给出核心答案
2. 使用证据支持观点（如：根据【XX章节】描述...）
3. 如信息不足，请说明："根据现有资料，无法确认..."
4. 保持中文回答，语言风格专业但不失亲和力

请开始回答：
"""

constructed_prompt = build_rag_prompt(user_query, context_chunks)

# # 调用大模型api，以deepseek为例
# from openai import OpenAI
# client = OpenAI(api_key="YOUR API KEY", base_url="https://api.deepseek.com")
#
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "你是一位资深小说读者，擅长根据文本证据进行严谨分析。回答必须基于提供的上下文，严禁编造信息。"},  # 可选的系统消息
#         {"role": "user", "content": constructed_prompt}
#     ],
#     temperature=0.3  # 降低随机性，使回答更基于事实
# )
# answer = response.choices[0].message.content
# print(answer)



# 调用本地 Ollama 模型
import ollama
response = ollama.generate(
    model="myqwen",  # 本地已下载的模型名称
    prompt=constructed_prompt,
    options={
        "temperature": 0.3,  # 控制生成随机性
    }
)
answer = response["response"]
print(answer)