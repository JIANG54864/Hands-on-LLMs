# ===== 1. 读取小说文本 =====
from sentence_transformers import SentenceTransformer

with open("../workspace/斗破苍穹1-500.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ===== 2. 清洗 =====
clean_text = raw_text.strip()

# ===== 3. 分块（保留段落结构）=====
paragraphs = [p for p in clean_text.split('\n') if p.strip()]
CHUNK_SIZE = 500  # 每个块最大字符数
chunks = []
current_chunk = ""

for para in paragraphs:
    if len(current_chunk) + len(para) <= CHUNK_SIZE:
        current_chunk += para + "\n"
    else:
        if current_chunk:
            chunks.append(current_chunk.strip())
        current_chunk = para + "\n"
if current_chunk:
    chunks.append(current_chunk.strip())


# ===== 4. 生成嵌入向量 =====
model = SentenceTransformer('shibing624/text2vec-base-chinese')
embeddings = model.encode(chunks)

# ===== 5. 存储到向量库 =====
import chromadb
client = chromadb.PersistentClient()
collection = client.create_collection("doupocangqiong1-500")

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": "斗破苍穹1-500.txt"}] * len(chunks)
)
print(f"共分成了 {len(chunks)} 个文本块")
print("第一个文本块示例：", chunks[0][:200])  # 打印前200字查看内容
