# RAG Document Q&A System | RAG文档问答系统

A document Q&A system built with LangChain + DeepSeek + ChromaDB.  
基于 LangChain + DeepSeek + ChromaDB 构建的文档问答系统。

## Features | 功能

- Upload PDF documents | 上传PDF文档
- Automatic text chunking and embedding | 自动切片和向量化
- Semantic search with ChromaDB | 基于语义检索
- AI answers based on document content only | AI仅根据文档内容回答

## Tech Stack | 技术栈

- LangChain LCEL
- DeepSeek-V3
- ChromaDB
- HuggingFace Embeddings
- Python

## Quick Start | 快速开始

```bash
pip install -r requirements.txt
python rag_langchain.py
```

## Why This Project | 项目意义

RAG is one of the most widely adopted AI patterns in enterprise.
RAG是企业AI落地最广泛的技术方案之一。

This project demonstrates the full pipeline:
本项目展示了完整的RAG流程：

Load → Chunk → Embed → Retrieve → Generate
