# AI 工具调用 Agent（命令行版）
# 演示：智谱 glm-4-flash 的 Function Calling——模型自动决定调用哪个工具
# 运行：python agent.py
import os
import sys
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from tools import TOOLS, execute_tool

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    print("❌ 未找到 ZHIPUAI_API_KEY，请在 .env 中配置")
    sys.exit(1)

client = ZhipuAI(api_key=api_key)


def run_agent(question: str, max_steps: int = 6) -> str:
    """模型与工具的对话循环：模型决定 -> 执行工具 -> 结果回传 -> 生成回答"""
    messages = [
        {"role": "system", "content": "你是会调用工具的智能助手。需要查时间、算数、查天气时，务必调用对应工具；其他问题正常回答。"},
        {"role": "user", "content": question},
    ]
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # 模型要求调用工具
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": tool_calls})

            for tc in msg.tool_calls:
                result = execute_tool(tc.function.name, tc.function.arguments)
                print(f"  🛠 调用工具: {tc.function.name}({tc.function.arguments})")
                print(f"     → {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            # 模型直接回答（不再需要工具）
            return msg.content or "（模型未返回内容）"

    return "（达到最大工具调用次数）"


if __name__ == "__main__":
    print("🤖 AI 工具调用 Agent（支持时间/计算/天气，输入 exit 退出）")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit", "退出"):
            break
        if not q:
            continue
        print("🤔 思考中...")
        print("💬 " + run_agent(q))
    print("👋 再见！")
