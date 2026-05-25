from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# =====================
# Step 1：加载并切分文档
# =====================

print("正在加载文档...")
loader = PyPDFLoader("04020426张陆毕设论文定稿.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(pages)
print(f"文档切分完成，共 {len(chunks)} 块")

# =====================
# Step 2：向量化存入FAISS
# =====================

print("正在向量化...")
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="BAAI/bge-m3"
)

# 分批向量化，每批最多50块
batch_size = 50
vectorstore = None

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"正在向量化第 {i//batch_size + 1} 批，共 {len(batch)} 块...")
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)

print("全部向量化完成")



print("向量化完成")

# =====================
# Step 3：构建检索器
# =====================

# search_kwargs={"k": 3} 表示检索最相关的3块
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =====================
# Step 4：自定义Prompt
# =====================

prompt_template = """你是一个文档问答助手，请根据以下参考内容回答用户的问题。
如果参考内容中没有相关信息，请直接说"文档中没有找到相关内容"，不要编造答案。

参考内容：
{context}

用户问题：{question}

回答："""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# =====================
# Step 5：组装链
# =====================
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def format_docs(docs):

    return "\n\n".join([doc.page_content for doc in docs])

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =====================
# Step 6：测试问答
# =====================

def ask(question):
    print(f"\n问题：{question}")
    result = qa_chain.invoke(question)
    print(f"答案：{result}")

    # 单独检索一次获取来源
    source_docs = retriever.invoke(question)
    print(f"\n参考来源（共{len(source_docs)}块）:")
    for i, doc in enumerate(source_docs):
        print(f"  [{i+1}] 第{doc.metadata.get('page', '?')+1}页: {doc.page_content[:80]}...")

ask("这篇文档的作者姓名是什么？")
ask("文档中包含哪些研究方向？")
ask("作者的身份证号是多少？")
