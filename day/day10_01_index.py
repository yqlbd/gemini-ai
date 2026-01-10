import os
import glob
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer


# --- 复用之前的配置 (保持一致) ---
class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        return [e.tolist() for e in embeddings]

    def name(self):
        return "local_sentence_transformer_v2"


# 连接数据库
print("💾 连接 ChromaDB...")
client = chromadb.PersistentClient(path="rst/chroma_db")
# ⚠️ 注意：我们可以用同一个集合，也可以新建一个专门存技术文档的
collection = client.get_or_create_collection(
    name="pangdundun_memory",  # 这里我们继续往同一个脑子里塞知识
    embedding_function=LocalEmbeddingFunction(),
)


# --- 核心逻辑 1: 文本切片器 (Text Splitter) ---
def split_text(text, chunk_size=300, chunk_overlap=50):
    """
    手动实现滑动窗口切片
    chunk_size: 每个块大约多少字符
    chunk_overlap: 重叠多少字符 (防止上下文切断)
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        # 截取片段
        chunk = text[start:end]
        chunks.append(chunk)

        # 移动窗口 (步长 = 大小 - 重叠)
        # 这样下一次循环就会回退一点点，包含上一块的尾巴
        start += chunk_size - chunk_overlap

    return chunks


# --- 核心逻辑 2: 读取文件并处理 ---
def process_markdown_files(directory):
    # 找到目录下所有的 .md 文件
    files = glob.glob(os.path.join(directory, "*.md"))
    print(f"📂 发现 {len(files)} 个 Markdown 文件: {files}")

    total_chunks = 0

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.basename(file_path)
        print(f"🔪 正在切分文件: {filename} (长度: {len(content)} 字符)...")

        # 1. 切片
        chunks = split_text(content, chunk_size=500, chunk_overlap=100)

        # 2. 准备入库数据
        # Chroma 需要：ID列表, 文本列表, 元数据列表
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        # 3. 入库 (Upsert)
        print(f"🚀 正在将 {len(chunks)} 个碎片存入向量库...")
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    print(
        f"\n✅ 全部完成！共处理了 {len(files)} 个文件，存入了 {total_chunks} 条记忆碎片。"
    )


# --- 运行 ---
if __name__ == "__main__":
    # 假设你的文档都在 rst 文件夹里
    # 你可以把 architecture_design.md 放在这里面测试
    target_dir = "./rst"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"⚠️ 目录 {target_dir} 不存在，已自动创建。请把 .md 文件放进去再运行！")
    else:
        process_markdown_files(target_dir)
