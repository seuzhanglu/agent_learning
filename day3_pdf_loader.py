from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# =====================
# Part 1：加载PDF
# =====================

loader = PyPDFLoader("04020426张陆毕设论文定稿.pdf")
pages = loader.load()

print(f"PDF共 {len(pages)} 页")
print(f"\n第一页内容（前200字）:")
print(pages[0].page_content[:200])
print(f"\n第一页的元数据:")
print(pages[0].metadata)

# =====================
# Part 2：文档切分
# =====================

# 把所有页合并成一个文本
full_text = "\n".join([page.page_content for page in pages])
print(f"\n文档总字数: {len(full_text)}")

# 定义切分函数，方便后面实验
def split_and_show(chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    chunks = splitter.create_documents([full_text])
    print(f"\nchunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"切分结果：共 {len(chunks)} 块")
    print(f"第一块内容：{chunks[0].page_content[:100]}...")
    print(f"第一块字数：{len(chunks[0].page_content)}")
    return chunks

# 实验：用不同参数切分，观察结果
print("\n" + "="*50)
print("切分参数实验")
print("="*50)

chunks_200 = split_and_show(chunk_size=200, chunk_overlap=20)
chunks_500 = split_and_show(chunk_size=500, chunk_overlap=50)
chunks_1000 = split_and_show(chunk_size=1000, chunk_overlap=100)