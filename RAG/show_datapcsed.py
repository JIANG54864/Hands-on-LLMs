from chromadb import PersistentClient

client = PersistentClient(path="./chroma")  # 确保路径正确
collection = client.get_collection(name="doupocangqiong1-500")

# 获取最后5个文档
results = collection.get()
last_five_documents = results['documents'][-5:]

print("最后5个文档内容：")
for i, doc in enumerate(last_five_documents, 1):
    print(f"第 {i} 个文档：\n{doc}\n{'=' * 30}")


