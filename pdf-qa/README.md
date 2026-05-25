# 📄 科研文献智能问答系统

基于 RAG（检索增强生成）架构的 PDF 智能问答系统，支持上传学术论文或任意 PDF 文档，通过自然语言提问快速定位关键信息。

## 🚀 在线 Demo

[点击体验](https://agentlearning-mokf5xxwddvasnpfcayjez.streamlit.app/)

## 📌 项目背景

科研过程中需要大量阅读文献，传统方式需要逐页浏览才能找到关键内容。本系统上传论文后可直接提问，通过语义检索定位相关段落再由 LLM 生成回答，比关键词搜索准确，比直接问 LLM 可靠。

## ✨ 功能特性

- 📤 支持任意 PDF 文档上传
- 🔍 基于语义的智能检索，优于关键词匹配
- 💬 支持多轮对话，具备上下文记忆
- 📎 回答时展示参考来源片段，结果可溯源
- ⚡ 文档分批向量化，支持大文件处理

## 🛠 技术架构


用户上传 PDF
↓
文档切分（RecursiveCharacterTextSplitter）
↓
向量化（BAAI/bge-m3 Embedding 模型）
↓
存入 FAISS 向量数据库
↓
用户提问 → 语义检索 Top-K 片段
↓
片段 + 对话历史 → LLM 生成回答
↓
展示答案 + 参考来源

## 📦 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | LangChain |
| 向量数据库 | FAISS |
| Embedding | BAAI/bge-m3 |
| LLM | Qwen2.5-72B-Instruct |
| 前端界面 | Streamlit |
| 部署 | Streamlit Cloud |

## 🔧 本地运行

**1. 克隆项目**
```bash
git clone https://github.com/你的用户名/agent-learning.git
cd agent-learning/pdf-qa
```

**2. 安装依赖**
```bash
pip install -r requirements.txt
```

**3. 配置 API Key**

新建 `.env` 文件：

OPENAI_API_KEY=你的API Key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1

**4. 启动应用**
```bash
streamlit run app.py
```