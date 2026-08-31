# 个人资料库 RAG 智能问答（Streamlit Web 应用）
# 运行：streamlit run app.py
import os
import sys
import streamlit as st
from zhipuai import ZhipuAI
import chromadb
from dotenv import load_dotenv
from knowledge import DOCUMENTS

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
# 兼容本地 .env 与云端 Streamlit Secrets 两种配置方式
api_key = os.getenv("ZHIPUAI_API_KEY") or st.secrets.get("ZHIPUAI_API_KEY", "")

st.set_page_config(page_title="个人知识库 RAG 问答", page_icon="🤖", layout="centered")
st.title("🤖 个人资料库 RAG 智能问答")
st.caption("检索增强生成（RAG）演示：文档向量化 → 语义检索 → 大模型生成")

if not api_key:
    st.error("❌ 未找到 ZHIPUAI_API_KEY，请在 .env 或云端 Secrets 中配置密钥。")
    st.stop()

client = ZhipuAI(api_key=api_key)
# 使用余弦相似度（embedding-3 官方推荐，更适合中文语义检索）
collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(
    name="my_docs", metadata={"hnsw:space": "cosine"}
)


def embed_texts(texts):
    """调用智谱 Embedding API 将文本转为向量"""
    response = client.embeddings.create(model="embedding-3", input=texts)
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


# 空库自动初始化（云端部署时 chroma_db 不会上传，首次启动自动建库）
if collection.count() == 0:
    with st.spinner("📚 首次运行：正在初始化知识库..."):
        seed_embs = embed_texts(DOCUMENTS)
        collection.upsert(
            documents=DOCUMENTS,
            embeddings=seed_embs,
            ids=[f"doc_{i}" for i in range(len(DOCUMENTS))],
        )
    st.success(f"✅ 知识库初始化完成，已存入 {len(DOCUMENTS)} 篇文档")


question = st.text_input("💬 输入你的问题：", placeholder="例如：我做过什么项目？")
ask = st.button("🚀 检索并回答", type="primary")

if ask and question.strip():
    # 1. 向量化问题并检索
    with st.spinner("🔍 正在检索资料..."):
        q_emb = embed_texts([question])[0]
        results = collection.query(query_embeddings=[q_emb], n_results=2)
    docs = results["documents"][0]
    distances = results["distances"][0]

    st.subheader("📂 检索到的相关资料")
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        st.info(f"**资料 {i + 1}**（相似度距离 {dist:.4f}）\n\n{doc}")

    # 2. 组装提示词并生成答案
    context = "\n".join(f"- {d}" for d in docs)
    prompt = (
        "你是基于检索资料的问答助手。请只依据下面给出的资料回答问题，"
        "不要编造资料中没有的内容，用简洁的中文回答。\n\n"
        f"【资料】\n{context}\n\n"
        f"【问题】{question}"
    )
    with st.spinner("🤖 大模型正在生成答案..."):
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        answer = response.choices[0].message.content

    st.subheader("💡 最终答案")
    st.write(answer)
    st.caption("提示：答案仅基于上方检索到的资料生成，可用于检查 RAG 效果。")

elif ask:
    st.warning("⚠️ 请输入问题后再点击检索。")
