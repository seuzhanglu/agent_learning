from dotenv import load_dotenv
import os
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(  # type: ignore[call-arg]
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    # 防止小模型在回答后继续幻想下一轮用户消息
    model_kwargs={"stop": ["用户：", "用户:", "\nuser\n", "\nUser\n", "Human:"]}
)

# =====================
# 方式一：没有Memory，LLM不记得上文
# =====================
print("=== 没有Memory ===")

def chat_no_memory(user_input):
    response = llm.invoke([HumanMessage(content=user_input)])
    print(f"用户：{user_input}")
    print(f"AI：{response.content}")
    print()

chat_no_memory("我叫张三，我是电磁场专业的研究生")
chat_no_memory("我叫什么名字？")   # LLM不记得，无法回答

# =====================
# 方式二：手动维护历史，LLM能记住
# =====================
print("=== 有Memory（手动维护）===")

messages = []

def chat_with_memory(user_input):
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"用户：{user_input}")
    print(f"AI：{response.content}")
    print(f"当前历史条数：{len(messages)}")
    print()

chat_with_memory("我叫张三，我是电磁场专业的研究生")
chat_with_memory("我叫什么名字？")     # 这次能记住
chat_with_memory("我的专业是什么？")   # 也能记住

# =====================
# 方式三：窗口记忆，只保留最近N轮
# =====================
print("=== 窗口记忆（只保留最近2轮）===")

from collections import deque

# maxlen=4 表示最多4条消息，即2轮对话
window = deque(maxlen=4)

def chat_window(user_input):
    window.append(HumanMessage(content=user_input))
    response = llm.invoke(list(window))
    window.append(AIMessage(content=response.content))
    print(f"用户：{user_input}")
    print(f"AI：{response.content}")
    print(f"窗口消息数：{len(window)}")
    print()

chat_window("我最喜欢的颜色是蓝色")
chat_window("我最喜欢的食物是火锅")
chat_window("我最喜欢的运动是篮球")   # 加入后，颜色那轮被挤出窗口
chat_window("我最喜欢的颜色是什么？")  # 应该不记得了，因为被挤出去了
chat_window("我最喜欢的运动是什么？")  # 应该记得，还在窗口里