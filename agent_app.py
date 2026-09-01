# AI 工具调用 Agent（Streamlit 网页版）
# 演示：Function Calling 多轮对话 + 工具调用过程可视化
# 运行：streamlit run agent_app.py
import os
import sys
import streamlit as st
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from tools import TOOLS, execute_tool

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY") or st.secrets.get("ZHIPUAI_API_KEY", "")

st.set_page_config(page_title="AI 工具调用 Agent", page_icon="🤖", layout="centered")
st.title("🤖 AI 工具调用 Agent")
st.caption("Function Calling 演示：模型自动决定调用 时间/计算/天气 工具，支持多轮对话")

if not api_key:
    st.error("❌ 未找到 ZHIPUAI_API_KEY，请在 .env 或云端 Secrets 中配置密钥。")
    st.stop()

client = ZhipuAI(api_key=api_key)

SYSTEM_PROMPT = "你是会调用工具的智能助手。需要查时间、算数、查天气时，务必调用对应工具；其他问题正常回答。"

# 会话历史（含 system 提示）
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def agent_loop(messages, max_steps=6):
    """在完整消息历史上运行工具循环，返回 (最终回答, 工具调用日志)"""
    msgs = list(messages)
    tool_logs = []
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
            msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls})
            for tc in msg.tool_calls:
                result = execute_tool(tc.function.name, tc.function.arguments)
                tool_logs.append((tc.function.name, tc.function.arguments, result))
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return msg.content or "（模型未返回内容）", tool_logs
    return "（达到最大工具调用次数）", tool_logs


# 渲染历史消息
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 输入框
if prompt := st.chat_input("问点什么，比如：广州今天天气怎么样？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            answer, tool_logs = agent_loop(st.session_state.messages)
        for name, args, result in tool_logs:
            with st.expander(f"🛠 调用工具：{name}({args})", expanded=False):
                st.code(result)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.sidebar.markdown("### 示例问题")
for ex in ["现在几点了？", "帮我算一下 (15+5)*3", "北京今天天气怎么样？"]:
    if st.sidebar.button(ex, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": ex})
        st.rerun()
if st.sidebar.button("🗑 清空对话", use_container_width=True):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.rerun()
