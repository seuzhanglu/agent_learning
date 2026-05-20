# day1_hello.py
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 加载.env文件里的环境变量
load_dotenv()

# 初始化模型
# 如果用国内中转，base_url会自动从.env读取
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 第一次对话：直接发消息
print("=== 测试1：直接对话 ===")
messages = [
    SystemMessage(content="你是一个helpful的助手，用中文回答。"),
    HumanMessage(content="用一句话解释什么是人工智能。")
]
response = llm.invoke(messages)
print(response.content)
print()

# 第二次：理解消息结构
print("=== 测试2：查看完整响应结构 ===")
print(f"回复类型: {type(response)}")
print(f"模型名称: {response.response_metadata.get('model_name', 'unknown')}")
print(f"消耗tokens: {response.usage_metadata}")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 测试3：PromptTemplate 传入变量
print("=== 测试3：PromptTemplate ===")
prompt = PromptTemplate(
    input_variables=["topic", "style"],
    template="用{style}的风格，用一句话解释{topic}是什么。"
)

# 用 | 把 prompt、llm、parser 串成一条链
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "机器学习", "style": "幽默"})
print(result)
print()

# 测试4：同一条链，换不同输入
print("=== 测试4：复用同一条链 ===")
topics = [
    {"topic": "神经网络", "style": "严肃学术"},
    {"topic": "深度学习", "style": "给小学生解释"},
]
for t in topics:
    print(f"[{t['style']}] {chain.invoke(t)}")