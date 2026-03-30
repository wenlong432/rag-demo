import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 初始化
client = OpenAI(
    api_key="sk-ijckjoaeslccrvhmpdgahcsbowqldlusnxmexxexemzjccqa",
    base_url="https://api.siliconflow.cn/v1"
)

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("documents")

def load_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def add_to_db(chunks):
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"✅ 已存储 {len(chunks)} 个文本片段")

def search(query, n=3):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n)
    return results["documents"][0]

def ask(query):
    relevant_chunks = search(query)
    context = "\n\n".join(relevant_chunks)
    prompt = f"""根据以下文档内容回答问题，如果文档中没有相关信息，请说"文档中未找到相关信息"。

【文档内容】
{context}

【问题】
{query}
"""
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 主流程
pdf_path = input("请输入PDF文件路径：")
print("读取PDF中...")
text = load_pdf(pdf_path)
chunks = split_text(text)
add_to_db(chunks)

print("\n✅ 文档加载完成，可以开始提问了！\n")

while True:
    query = input("你的问题：")
    if query == "quit":
        break
    print(f"\nAI：{ask(query)}\n")