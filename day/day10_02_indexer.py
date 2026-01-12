import os
import glob
import json
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer


# --- 1. 复用 Embedding 配置 ---
class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # 既然是处理文档，我们用这一款支持多语言的模型，效果比较均衡
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        # 列表推导式，把每个 numpy array 转成 list
        return [e.tolist() for e in embeddings]

    def name(self):
        return "local_sentence_transformer_v2"


# 连接数据库，定义好数据库地址，这里为rst/chroma_db
print("💾 正在连接记忆库...")
client = chromadb.PersistentClient(path="rst/chroma_db")

# ⚠️ 注意：为了演示效果，我们这次创建一个全新的集合，叫 "categorized_memory" (分类记忆)，相当于表名
# 这样不会和之前的混乱数据混在一起
collection = client.get_or_create_collection(
    name="categorized_memory", embedding_function=LocalEmbeddingFunction()
)


# --- 2. 核心工具：切片器 ---
"""
text: 待切片的文本
chunk_size: 每个切片的最大长度，默认300字符
chunk_overlap: 切片之间重叠的部分，保证上下文连贯，默认50字符
"""


def split_text(text, chunk_size=300, chunk_overlap=50) -> list:
    """滑动窗口切片：保证上下文连贯"""
    chunks = []
    # 游标位置
    start = 0
    # 切片文本长度
    text_len = len(text)

    # 当游标没到文本末尾时，循环持续切片
    while start < text_len:
        # 切片结束位置
        end = start + chunk_size
        # 切片文本，取游标到结束位置之间的内容
        chunk = text[start:end]
        # 加入集合
        chunks.append(chunk)
        # 步长 = 窗口大小 - 重叠部分
        start += chunk_size - chunk_overlap
    return chunks


# --- 3. 处理不同类型的文件 ---
# 处理*.md 技术文档，打上 category: tech 标签
def process_tech_docs(directory) -> None:
    """处理技术文档 (.md) -> 打上 category: tech"""
    files = glob.glob(os.path.join(directory, "*.md"))
    print(f"\n📘 发现 {len(files)} 个技术文档")

    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 切片，每片400字符，重叠50字符
        chunks = split_text(content, chunk_size=400, chunk_overlap=50)

        # 准备入库数据
        ids = [f"tech_{filename}_{i}" for i in range(len(chunks))]

        # 🔥 关键步骤：打标签！
        # 我们明确指定这是 "tech" 类，作者是 "赵一清"
        # 列表推导式生成 metadatas
        metadatas = [
            {
                "category": "tech",
                "author": "赵一清",
                "source": filename,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        print(f"   ↳ 正在存入 '{filename}': {len(chunks)} 个碎片 (Tag: tech)")
        # 入库，每次读到.md 文件就存一批
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)


# 处理*.json 日记文件，打上 category: diary 标签
def process_diary_logs(directory):
    """处理日记文件 (.json) -> 打上 category: diary"""
    files = glob.glob(os.path.join(directory, "*.json"))
    print(f"\nCb 发现 {len(files)} 个日记文件")

    for file_path in files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)  # 假设是 list of dict
                # 判断是列表还是字典，如果是[]就继续，同理，{}是dict
                if not isinstance(data, list):
                    print(f"   ⚠️ 跳过 {filename}: 格式不是列表")
                    continue
            except:
                print(f"   ⚠️ 跳过 {filename}: JSON 解析失败")
                continue

        chunks = []
        ids = []
        metadatas = []

        # 同时获取角标和对象的写法，从0开始，entry是个dict对象，直接可以get字段
        for i, entry in enumerate(data):
            # 把 JSON 对象转成这种易读的字符串
            # 假设 entry 长这样: {"timestamp": "...", "event": "..."}
            text_chunk = f"时间: {entry.get('timestamp', '未知')}\n事件: {entry.get('event', str(entry))}"

            chunks.append(text_chunk)
            ids.append(f"diary_{filename}_{i}")

            # 🔥 关键步骤：打标签！
            # 明确这是 "diary" 类，主角是 "胖墩墩"
            metadatas.append(
                {"category": "diary", "subject": "胖墩墩", "source": filename}
            )

        if chunks:
            print(f"   ↳ 正在存入 '{filename}': {len(chunks)} 条记录 (Tag: diary)")
            collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)


# --- 运行主程序 ---
if __name__ == "__main__":
    target_dir = "./rst"  # 你的文件都在这里

    # 1. 清空旧数据 (为了演示纯净的效果)
    # collection.delete(where={}) # 如果你想追加而不是覆盖，就把这行注释掉

    # 2. 分类处理
    process_tech_docs(target_dir)
    process_diary_logs(target_dir)

    print("\n✅ 索引重建完成！数据已打上 Metadata 标签。")
    print("现在 Gemini 不会再把胖墩墩当成架构师了！🐶")
