import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# 1. 初始化本地模型
print("🤖 正在加载本地语义引擎 (paraphrase-multilingual-MiniLM-L12-v2)...")
local_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


class LocalEmbeddingFunction(EmbeddingFunction):
    """
    修复版：继承官方基类，并实现 name() 方法
    """

    def __call__(self, input: Documents) -> Embeddings:
        # encode 方法默认返回 numpy array，Chroma 需要 list
        embeddings = local_model.encode(input)
        return [e.tolist() for e in embeddings]

    def name(self) -> str:
        # ✅ 关键修复：ChromaDB 需要这个方法来验证模型身份
        return "local_sentence_transformer_v2"


def main():
    print("📦 正在初始化 ChromaDB 数据库 (本地持久化)...")

    # 2. 初始化持久化客户端
    client = chromadb.PersistentClient(path="rst/chroma_db")

    # 3. 创建记忆集合
    collection = client.get_or_create_collection(
        name="pangdundun_memory", embedding_function=LocalEmbeddingFunction()
    )

    # 4. 准备一些“非结构化”的日记数据
    documents = [
        "胖墩墩最爱吃的是水煮鸡胸肉，每次看到都会流口水。",  # doc1
        "胖墩墩非常讨厌洗澡，每次去宠物店都要躲在床底下。",  # doc2
        "今天天气不错，赵一清（宝）带胖墩墩在草地上接到了三次飞盘。",  # doc3
        "胖墩墩的生日是2023年5月20日，是个金牛座宝宝，性格很倔强。",  # doc4
    ]

    ids = ["diary_001", "diary_002", "diary_003", "diary_004"]

    metadatas = [
        {"type": "diet", "date": "2026-01-10"},
        {"type": "habit", "date": "2026-01-11"},
        {"type": "activity", "date": "2026-01-12"},
        {"type": "info", "date": "2023-05-20"},
    ]

    print("✍️ 正在写入记忆数据 (Upsert)...")
    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    print(f"✅ 成功存储了 {collection.count()} 条记忆！")

    # 5. 见证奇迹时刻：语义搜索
    print("\n🔍 开始意图匹配测试 (RAG 的核心)：")
    print("-" * 50)

    test_queries = [
        "狗狗喜欢吃什么食物？",
        "它害怕什么事情？",
        "这只狗多大了？",
        "Python编程语言怎么学？",
    ]

    # 设定一个阈值（经过观察，25 以下的才是靠谱的）
    # 注意：Chroma 默认的 L2 距离是越小越好
    DISTANCE_THRESHOLD = 25.0

    for q in test_queries:
        print(f"\n❓ 提问: {q}")

        results = collection.query(query_texts=[q], n_results=1)

        if results["documents"] and results["documents"][0]:
            found_doc = results["documents"][0][0]
            distance = results["distances"][0][0]  # 获取距离

            # 🛑 核心逻辑：加了这层判断，AI 就不敢乱说了
            if distance < DISTANCE_THRESHOLD:
                print(f"💡 找到答案 (距离 {distance:.4f}):\n   👉 {found_doc}")
            else:
                print(
                    f"❌ 未找到相关结果 (最近匹配距离 {distance:.4f} > 阈值 {DISTANCE_THRESHOLD})"
                )
        else:
            print("❌ 库里是空的")


if __name__ == "__main__":
    main()
