from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# =====================
# Step 1：定义工具（和上面一样）
# =====================

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。
    当用户询问某个城市的天气、温度、气候时使用此工具。
    
    Args:
        city: 城市名称，例如：北京、上海、广州
    """
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
    
    Args:
        expression: 数学表达式，例如：2+3*4、100/5、2**10
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """
    搜索知识库中的信息。
    当用户询问编程语言、技术概念、框架相关问题时使用此工具。
    
    Args:
        query: 搜索关键词
    """
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


# =====================
# Step 2：初始化模型，绑定工具
# =====================

tools = [get_weather, calculate, search_knowledge]
tools_map = {tool.name: tool for tool in tools}  # 方便后面按名字调用

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 把工具绑定到模型，LLM才知道有哪些工具可以用
llm_with_tools = llm.bind_tools(tools)

# =====================
# Step 3：实现 ReAct 循环
# =====================

def run_agent(user_input: str, verbose: bool = True):
    """
    运行Agent，循环执行直到LLM不再调用工具为止
    verbose=True 时打印每一步的推理过程
    """
    messages = [
        SystemMessage(content=(
            "你是一个严格遵守工具调用规则的智能助手。\n\n"
            "你拥有以下工具，遇到对应场景时【必须调用工具】，严禁用自己的知识直接回答：\n"
            "- get_weather(city)：查询城市实时天气。只要用户提到任何城市的天气、温度、气候，必须调用此工具。\n"
            "- calculate(expression)：计算数学表达式。用户需要任何数学计算时，必须调用此工具；"
            "若用户用中文描述算式（如'2的10次方加上3乘以50'），先将其转换为Python表达式（如2**10+3*50），再调用工具。\n"
            "- search_knowledge(query)：搜索技术知识库。用户询问编程语言、技术框架、技术概念时，必须调用此工具。\n\n"
            "【禁止行为】不得在未调用工具的情况下直接回答天气、数学计算、技术知识类问题。\n"
            "【允许直接回答】仅限日常问候、自我介绍等与上述三类工具完全无关的话题。"
        )),
        HumanMessage(content=user_input),
    ]

    if verbose:
        print(f"\n{'='*50}")
        print(f"用户输入：{user_input}")
        print(f"{'='*50}")
    
    # ReAct循环：最多执行5轮，防止无限循环
    for step in range(5):
        if verbose:
            print(f"\n--- 第{step+1}轮思考 ---")
        
        # LLM思考：需要调用工具吗？调用哪个？
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # 如果LLM没有调用工具，说明它认为可以直接回答了
        if not response.tool_calls:
            if verbose:
                print(f"LLM决定直接回答，不需要工具")
                print(f"[调试] tool_calls 为空，模型回答: {response.content[:120] if response.content else '(空)'}")
            break
        
        # LLM决定调用工具
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if verbose:
                print(f"调用工具：{tool_name}")
                print(f"参数：{tool_args}")
            
            # 执行工具
            tool_result = tools_map[tool_name].invoke(tool_args)
            
            if verbose:
                print(f"工具返回：{tool_result}")
            
            # 把工具结果加入消息历史，LLM下一轮会看到这个结果
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))
    
    final_answer = response.content
    print(f"\n最终答案：{final_answer}")
    return final_answer


# =====================
# Step 4：测试Agent
# =====================

# 测试1：需要调用天气工具
run_agent("北京今天天气怎么样？")

# 测试2：需要调用计算工具
run_agent("帮我计算一下 2的10次方 加上 3乘以50 等于多少")

# 测试3：需要调用知识库工具
run_agent("LangChain是什么？")

# 测试4：不需要工具，直接回答
run_agent("你好，介绍一下你自己")

# 测试5：需要同时调用多个工具
run_agent("北京和上海的天气分别是什么？哪个城市更适合出行？")