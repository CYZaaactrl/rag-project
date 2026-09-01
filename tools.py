# AI Agent 工具集：工具实现 + Function Calling Schema
# 命令行版（agent.py）和网页版（agent_app.py）共用

import json
import ast
import datetime
import urllib.request
import urllib.parse


# ============ 工具实现 ============

def get_current_time():
    """获取当前日期和时间"""
    now = datetime.datetime.now()
    return f"当前时间是 {now.strftime('%Y年%m月%d日 %H:%M:%S')}"


def calculate(expression: str):
    """安全计算数学表达式（ast 白名单校验，防止代码注入）"""
    try:
        tree = ast.parse(expression, mode="eval")
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return "错误：表达式包含不允许的内容"
        return f"计算结果：{eval(compile(tree, '', 'eval'))}"
    except Exception as e:
        return f"计算失败：{e}"


def get_weather(city: str):
    """查询城市当前天气（免费 wttr.in 接口）"""
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1&lang=zh"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cur = data["current_condition"][0]
        desc = cur["weatherDesc"][0]["value"]
        return f"{city}当前天气：{desc}，温度 {cur['temp_C']}°C，体感 {cur['FeelsLikeC']}°C，湿度 {cur['humidity']}%"
    except Exception as e:
        return f"天气查询失败：{e}（可能是网络受限）"


# ============ 工具注册表 ============

TOOL_MAP = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
}

# Function Calling Schema：声明给模型看的工具描述
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持四则运算和括号，如 2+3*4、(10-2)/2",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 北京、广州"}
                },
                "required": ["city"],
            },
        },
    },
]


def execute_tool(name: str, arguments: str) -> str:
    """根据模型返回的工具调用，执行对应的 Python 函数"""
    args = json.loads(arguments) if arguments else {}
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"未知工具：{name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"工具参数错误：{e}"
