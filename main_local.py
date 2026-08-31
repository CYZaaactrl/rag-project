from sentence_transformers import SentenceTransformer
import chromadb

print("🚀 开始运行RAG示例...")

# 1. 加载嵌入模型（首次运行会自动下载模型，约90MB）
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ 模型加载成功")

# 2. 创建向量数据库
client = chromadb.PersistentClient(path="./my_db")
collection = client.get_or_create_collection(name="my_docs")
print("✅ 数据库连接成功")

# 3. 准备示例文档
documents = [
    "我是一名机械电子工程专业的学生，熟悉Python和嵌入式开发。",
    "我做过RAG项目，用LangChain和Chroma实现了文档问答。",
    "我对AI Agent感兴趣，用Claude Code辅助过树莓派开发。"
]

# 4. 向量化并存入
embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=["doc1", "doc2", "doc3"]
)
print(f"✅ 已存入 {len(documents)} 篇文档")

# 5. 用户提问
question = "你做过什么项目？"
print(f"📝 用户问题: {question}")

# 6. 检索
question_emb = model.encode([question]).tolist()
results = collection.query(query_embeddings=question_emb, n_results=2)

# 7. 输出结果
print("\n🔍 检索到的相关内容：")
for i, doc in enumerate(results['documents'][0]):
    print(f"  {i+1}. {doc}")

print("\n✅ RAG示例运行完成！")
