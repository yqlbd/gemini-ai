import chromadb

# 引入上面的 Embedding 类 (为了省事，实际项目中应该单独放一个文件)
from day10_02_indexer import LocalEmbeddingFunction

client = chromadb.PersistentClient(path="rst/chroma_db")
collection = client.get_collection(
    name="categorized_memory", embedding_function=LocalEmbeddingFunction()
)


def ask_with_filter(question, category_filter):
    print(f"\n❓ 问题: {question}")
    print(f"🔍 过滤器: category == '{category_filter}'")

    results = collection.query(
        query_texts=[question],
        n_results=3,
        # 🔥 核心魔法：Where 子句
        # 告诉 Chroma 只在指定的 category 里找
        where={"category": category_filter},
    )

    if results["documents"][0]:
        print(f"✅ 找到答案 (Metadata: {results['metadatas'][0][0]}):")
        print(f"   👉 {results['documents'][0][0][:50]}...")  # 只打印前50个字
    else:
        print("❌ 没找到相关信息 (因为被过滤器拦住了)")


# --- 测试 ---
# 1. 问胖墩墩会不会写代码 (强制查日记)
ask_with_filter("会写代码吗？", category_filter="diary")

# 2. 问赵一清会不会写代码 (强制查技术文档)
ask_with_filter("会写代码吗？", category_filter="tech")
