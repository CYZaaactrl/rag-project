import os
import sys
from zhipuai import ZhipuAI
import chromadb
from dotenv import load_dotenv
from knowledge import DOCUMENTS

# Windows 控制台默认 GBK 编码，无法打印 emoji，强制切换为 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 从 .env 文件加载 API Key
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")

if not api_key:
    print("❌ 错误：未找到 ZHIPUAI_API_KEY")
    print("请确保 .env 文件存在且包含：ZHIPUAI_API_KEY=你的密钥")
    exit(1)

print("🚀 开始运行RAG示例（使用智谱 Embedding API）...")

# 1. 初始化智谱客户端
client = ZhipuAI(api_key=api_key)

# 2. 定义向量化函数
def embed_texts(texts):
    """调用智谱 Embedding API，将文本转为向量"""
    response = client.embeddings.create(
        model="embedding-3",
        input=texts,
    )
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]

# 3. 创建向量数据库（余弦相似度，embedding-3 官方推荐）
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(
    name="my_docs", metadata={"hnsw:space": "cosine"}
)
print("✅ 数据库连接成功")

# 4. 准备文档（来自 knowledge.py）
documents = DOCUMENTS

# 5. 向量化并存入
print("🔄 正在向量化文档...")
embeddings = embed_texts(documents)
collection.upsert(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)
print(f"✅ 已存入 {len(documents)} 篇文档")

# 6. 用户提问
question = "你做过什么项目？"
print(f"📝 用户问题: {question}")

# 7. 向量化问题并检索
print("🔄 正在向量化问题...")
question_emb = embed_texts([question])[0]
results = collection.query(
    query_embeddings=[question_emb],
    n_results=2
)

# 8. 输出检索结果
print("\n🔍 检索到的相关内容：")
for i, doc in enumerate(results['documents'][0]):
    print(f"  {i+1}. {doc}")

# 9. 组装提示词并调用 glm-4-flash 生成最终答案
print("\n🤖 正在调用 glm-4-flash 生成答案...")
retrieved_docs = results['documents'][0]
context = "\n".join(f"- {doc}" for doc in retrieved_docs)
prompt = (
    "你是基于检索资料的问答助手。请只依据下面给出的资料回答问题，"
    "不要编造资料中没有的内容，用简洁的中文回答。\n\n"
    f"【资料】\n{context}\n\n"
    f"【问题】{question}"
)
response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)
answer = response.choices[0].message.content

print(f"\n💡 最终答案：\n{answer}")
print("\n✅ RAG示例运行完成！")
