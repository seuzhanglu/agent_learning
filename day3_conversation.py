from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# 加载文档（和上面一样）
loader = PyPDFLoader("04020426张陆毕设论文定稿.pdf")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="BAAI/bge-m3",
    chunk_size=64
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# =====================
# 关键：加入Memory
# k=5 表示记住最近5轮对话
# =====================
# 用字典存每个会话的历史记录
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 新版多轮对话Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个文档问答助手，根据以下参考内容回答问题。
如果参考内容中没有相关信息，请说"文档中没有找到相关内容"，不要编造答案。

参考内容：{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 组装链
chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x.get("chat_history", [])
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 包装成带历史记录的链
qa_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# =====================
# 模拟多轮对话
# =====================

def chat(question, session_id="default"):
    print(f"\n用户：{question}")
    result = qa_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"助手：{result}")
    return result

print("开始多轮对话测试（根据你的PDF内容修改问题）")
print("="*50)

# 第一轮：问一个基础问题
chat("这篇文档主要讲了什么？")

# 第二轮：追问，测试它能否记住上下文
chat("你刚才说的第一点能详细解释一下吗？")

# 第三轮：继续追问
chat("这和我们平时理解的有什么不同？")