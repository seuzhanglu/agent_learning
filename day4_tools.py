from dotenv import load_dotenv
import os
from langchain_core.tools import tool

load_dotenv()

# =====================
# 用 @tool 装饰器定义工具
# 注意：docstring（三引号注释）非常重要
# LLM 靠读 docstring 来决定什么时候用这个工具
# =====================

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。
    当用户询问某个城市的天气、温度、气候时使用此工具。
    
    Args:
        city: 城市名称，例如：北京、上海、广州
    """
    # 模拟天气数据，实际项目里会调用真实天气API
    weather_data = {
        "北京": "晴天，温度25°C，湿度40%",
        "上海": "多云，温度28°C，湿度65%",
        "广州": "小雨，温度32°C，湿度80%",
        "深圳": "阴天，温度30°C，湿度75%",
    }
    return weather_data.get(city, f"暂无{city}的天气数据")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式的结果。
    当用户需要进行数学计算时使用此工具。
    支持加减乘除、幂运算等基本运算。
    
    Args:
        expression: 数学表达式，例如：2+3*4、100/5、2**10
    """
    try:
        # eval执行数学表达式，只允许基本运算
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """
    搜索知识库中的信息。
    当用户询问编程语言、技术概念、框架相关的问题时使用此工具。
    
    Args:
        query: 搜索关键词
    """
    # 模拟知识库，实际项目里会换成真正的RAG检索
    knowledge_base = {
        "python": "Python是一种简洁易学的编程语言，广泛用于数据科学、AI和Web开发。",
        "java": "Java是面向对象的编程语言，以跨平台特性著称，广泛用于企业级开发。",
        "langchain": "LangChain是构建LLM应用的框架，提供链、Agent、工具等核心组件。",
        "rag": "RAG是检索增强生成技术，通过检索相关文档来增强LLM的回答质量。",
        "spring boot": "Spring Boot是Java最流行的Web开发框架，简化了Spring应用的配置和部署。",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    return f"知识库中没有找到关于'{query}'的信息"


# 测试工具是否正常工作
print("=== 测试工具 ===")
print(get_weather.invoke("北京"))
print(calculate.invoke("2**10"))
print(search_knowledge.invoke("什么是RAG"))

# 查看工具的元数据（LLM会看这些来决定用哪个工具）
print("\n=== 工具元数据 ===")
print(f"工具名称: {get_weather.name}")
print(f"工具描述: {get_weather.description}")