from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 初始化LLM
llm = ChatOpenAI(
    api_key="sk-ijckjoaeslccrvhmpdgahcsbowqldlusnxmexxexemzjccqa",
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3"
)

# 初始化embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 加载PDF
pdf_path = input("请输入PDF文件路径：")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 切片
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"✅ 文档切分为 {len(chunks)} 个片段")

# 存入向量数据库
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ 向量数据库构建完成")

# 构建prompt
prompt = ChatPromptTemplate.from_template("""
根据以下文档内容回答问题，如果文档中没有相关信息，请说"文档中未找到相关信息"。

【文档内容】
{context}

【问题】
{question}
""")

# 构建chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print("\n✅ 准备完成，开始提问！\n")

while True:
    query = input("你的问题：")
    if query == "quit":
        break
    result = chain.invoke(query)
    print(f"\nAI：{result.content}\n")