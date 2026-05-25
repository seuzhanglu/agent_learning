from dotenv import load_dotenv
import os
import streamlit as st
from pydantic import SecretStr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import tempfile

load_dotenv()

# 页面配置，必须是第一个Streamlit命令
st.set_page_config(
    page_title="PDF 智能问答",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF 智能问答系统")
st.caption("上传 PDF 文档，然后用自然语言提问")

# =====================
# 初始化模型（加缓存，只初始化一次）
# =====================
@st.cache_resource
def init_models():
    embeddings = OpenAIEmbeddings(
        api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model="BAAI/bge-m3"
    )
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-72B-Instruct",
        temperature=0,
        api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    return embeddings, llm

embeddings, llm = init_models()

# =====================
# session_state 初始化
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# =====================
# 侧边栏：上传文档
# =====================
with st.sidebar:
    st.header("📁 文档管理")

    uploaded_file = st.file_uploader(
        "上传 PDF 文件",
        type=["pdf"]
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.doc_name:
            with st.spinner("正在处理文档..."):

                # 保存到临时文件
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # 加载切分
                loader = PyMuPDFLoader(tmp_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(pages)

                # 分批向量化
                vectorstore = None
                batch_size = 50
                progress = st.progress(0)

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(
                            batch, embeddings
                        )
                    else:
                        vectorstore.add_documents(batch)
                    progress.progress(
                        min((i + batch_size) / len(chunks), 1.0)
                    )

                st.session_state.vectorstore = vectorstore
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []

            st.success("✅ 文档处理完成")
            st.info(f"共切分为 {len(chunks)} 个片段")

    if st.session_state.doc_name:
        st.divider()
        st.caption(f"当前文档：{st.session_state.doc_name}")
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()

# =====================
# 主界面：对话区域
# =====================
if st.session_state.vectorstore is None:
    st.info("👈 请先在左侧上传 PDF 文档")
else:
    # 显示历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 输入框
    if question := st.chat_input("输入你的问题..."):

        with st.chat_message("user"):
            st.write(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):

                # 检索相关文档
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join(
                    [doc.page_content for doc in relevant_docs]
                )

                # 构建消息列表（系统提示 + 历史 + 当前问题）
                # Make the list untyped so we can append different Message types
                messages_for_llm: list = [
                    SystemMessage(content=f"""你是一个文档问答助手。
根据以下参考内容回答用户问题。
如果参考内容中没有相关信息，请说"文档中没有找到相关内容"。

参考内容：
{context}""")
                ]

                # 加入最近10条历史（5轮对话）
                for msg in st.session_state.messages[-10:]:
                    if msg["role"] == "user":
                        messages_for_llm.append(
                            HumanMessage(content=msg["content"])
                        )
                    else:
                        messages_for_llm.append(
                            AIMessage(content=msg["content"])
                        )

                response = llm.invoke(messages_for_llm)
                answer = response.content

            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # 折叠显示参考来源
        with st.expander("查看参考来源"):
            for i, doc in enumerate(relevant_docs):
                st.caption(
                    f"片段 {i+1}（第 {doc.metadata.get('page', '?')+1} 页）"
                )
                st.text(doc.page_content[:200] + "...")
                st.divider()