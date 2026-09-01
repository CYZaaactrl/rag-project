# 🔗 线上作品链接

> 投简历 / 面试时使用的线上作品集入口

## 🤖 AI 工具调用 Agent
- **地址**：https://rag-project-ftj9u2qnw4b7rtb58pdkau.streamlit.app/
- **说明**：Function Calling 工具调用演示（时间 / 计算 / 天气），多轮对话 + 工具过程可视化
- **状态**：✅ 已验证可公开访问（HTTP 200，无需登录）

## 📚 RAG 智能问答系统
- **地址**：https://rag-project-hzf9yejmbheealsjbmtote.streamlit.app/
- **说明**：基于我的 12 份简历知识库 + 智谱 GLM 大模型的检索增强生成问答系统
- **状态**：✅ 已验证可公开访问（HTTP 200，无需登录）

---

## 📌 本地运行方式
```bash
# RAG 应用（本地 8501 端口）
streamlit run app.py

# Agent 应用（本地 8502 端口）
streamlit run agent_app.py
```

## 🗝️ 云端密钥
- 两个应用均需配置 `ZHIPUAI_API_KEY`（云端 Secrets 已配置，勿泄露）
