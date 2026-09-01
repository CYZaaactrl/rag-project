# 个人资料库 RAG 智能问答（Streamlit Web 应用）— 升级版
# 新增功能：文件上传（PDF/Word/TXT）、引用来源、Rerank 重排
# 运行：streamlit run app.py
import os
import sys
import io
import httpx
import chromadb
import streamlit as st
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from knowledge import DOCUMENTS

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
# 兼容本地 .env 与云端 Streamlit Secrets 两种配置方式
api_key = os.getenv("ZHIPUAI_API_KEY") or st.secrets.get("ZHIPUAI_API_KEY", "")

st.set_page_config(page_title="个人知识库 RAG 问答", page_icon="🤖", layout="centered")
st.title("🤖 个人资料库 RAG 智能问答")
st.caption("检索增强生成（RAG）：文档向量化 → 语义检索 → Rerank 重排 → 大模型生成")

if not api_key:
    st.error("❌ 未找到 ZHIPUAI_API_KEY，请在 .env 或云端 Secrets 中配置密钥。")
    st.stop()

client = ZhipuAI(api_key=api_key)
# 内置知识库（余弦相似度，embedding-3 官方推荐，更适合中文语义检索）
collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(
    name="my_docs", metadata={"hnsw:space": "cosine"}
)
# 上传文档库（独立集合，带 source 元数据记录文件名）
up_collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(
    name="uploaded_docs", metadata={"hnsw:space": "cosine"}
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


# ============ 文件解析与切块 ============

def parse_pdf(data):
    """用 pypdf 解析 PDF 文本"""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def parse_docx(data):
    """用 python-docx 解析 Word 文本"""
    from docx import Document
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def parse_file(filename, data):
    """按扩展名选择解析器"""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return parse_pdf(data)
    elif ext == "docx":
        return parse_docx(data)
    else:
        return data.decode("utf-8", errors="ignore")


def chunk_text(text, size=200, overlap=50):
    """按固定长度切块，带重叠，保留语义完整性"""
    text = " ".join(text.split())  # 合并空白与换行
    if len(text) <= size:
        return [text] if text else []
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


# ============ 检索 + Rerank ============

def rerank(query, items, top_n=3):
    """用智谱 rerank 对候选重排；失败时回退按原始相似度排序"""
    texts = [it[0] for it in items]
    try:
        resp = httpx.post(
            "https://open.bigmodel.cn/api/paas/v4/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "rerank", "query": query, "documents": texts, "top_n": top_n},
            timeout=30,
        )
        resp.raise_for_status()
        ranked = []
        for r in resp.json().get("results", []):
            idx = r["index"]
            ranked.append((items[idx][0], items[idx][1], r.get("relevance_score", 0)))
        return ranked
    except Exception:
        # 回退：按原始相似度降序取前 top_n
        return sorted(items, key=lambda x: x[2], reverse=True)[:top_n]


def search(query, top_k=6, use_rerank=True, rerank_top=3):
    """检索内置库 + 上传库，返回 [(文本, 来源, 相关度)]"""
    q_emb = embed_texts([query])[0]
    items = []

    # 内置知识库（无 source 元数据，来源固定为「内置资料库」）
    r = collection.query(query_embeddings=[q_emb], n_results=top_k)
    for doc, dist in zip(r["documents"][0], r["distances"][0]):
        items.append((doc, "📚 内置资料库", 1 - dist))  # 余弦距离转相似度

    # 上传文档库（带 source 元数据 = 文件名）
    if up_collection.count() > 0:
        r2 = up_collection.query(query_embeddings=[q_emb], n_results=top_k)
        metas = r2["metadatas"][0]
        for doc, meta, dist in zip(r2["documents"][0], metas, r2["distances"][0]):
            src = (meta or {}).get("source", "上传文档")
            items.append((doc, f"📄 {src}", 1 - dist))

    if not items:
        return []
    if use_rerank:
        return rerank(query, items, top_n=rerank_top)
    return sorted(items, key=lambda x: x[2], reverse=True)[:rerank_top]


# ============ 侧边栏：上传文档 + 设置 ============
st.sidebar.header("📥 上传文档到知识库")
uploaded_files = st.sidebar.file_uploader(
    "支持 PDF / Word / TXT（可多选）", type=["pdf", "docx", "txt"], accept_multiple_files=True
)
if st.sidebar.button("🚀 上传并建立索引", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("请先选择文件")
    else:
        with st.spinner("⏳ 正在解析并向量化..."):
            new_count = 0
            for f in uploaded_files:
                text = parse_file(f.name, f.getvalue())
                chunks = chunk_text(text)
                if not chunks:
                    st.sidebar.warning(f"「{f.name}」未解析出文字，已跳过")
                    continue
                base = f.name.rsplit(".", 1)[0].replace(" ", "_")
                ids = [f"up_{base}_{i}" for i in range(len(chunks))]
                up_collection.upsert(
                    documents=chunks,
                    embeddings=embed_texts(chunks),
                    metadatas=[{"source": f.name}] * len(chunks),
                    ids=ids,
                )
                new_count += len(chunks)
        st.sidebar.success(f"✅ 已加入 {new_count} 个文本片段")

use_rerank = st.sidebar.toggle("🔁 使用 Rerank 重排", value=True)
st.sidebar.caption("Rerank 提升检索精准度；失败时自动回退")
if st.sidebar.button("🗑 清空上传文档", use_container_width=True):
    ids = up_collection.get()["ids"]
    if ids:
        up_collection.delete(ids=ids)
    st.sidebar.success("已清空上传文档")


# ============ 主区：问答 ============
question = st.text_input("💬 输入你的问题：", placeholder="例如：我做过什么项目？")
ask = st.button("🚀 检索并回答", type="primary")

if ask and question.strip():
    with st.spinner("🔍 正在检索资料..."):
        top_items = search(question, use_rerank=use_rerank)

    if not top_items:
        st.warning("没有检索到相关资料，换个问法试试。")
    else:
        st.subheader("📂 检索到的相关资料（含来源）")
        for i, (text, source, score) in enumerate(top_items):
            st.info(f"**{source}**（相关度 {score:.4f}）\n\n{text}")

        # 组装提示词并生成答案（要求模型标注来源）
        context = "\n".join(f"- [{src}] {d}" for d, src, _ in top_items)
        prompt = (
            "你是基于检索资料的问答助手。请只依据下面给出的资料回答问题，"
            "并在回答末尾用「来源：」一行列出引用资料的来源标签。"
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
        st.caption("答案基于上方带来源的资料生成，来源标签供核对。")

elif ask:
    st.warning("⚠️ 请输入问题后再点击检索。")
