import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def local_embedding_test():
    print("🚀 正在加载本地模型 (paraphrase-multilingual-MiniLM-L12-v2)...")
    # 这个模型支持 50+ 种语言，包括中文，维度是 384 维
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 1. 定义完全不同的测试文本
    text_base = "胖墩墩"
    compare_list = [
        "一只可爱的小狗",
        "今天天气真好",
        "Python编程语言",
        "胖墩墩",  # 故意放一个完全一样的，看它是不是 1.0
    ]

    # 2. 生成向量 (本地计算)
    print(f"\n🧪 正在为 '{text_base}' 生成本地向量...")
    vec_base = model.encode([text_base])
    print(f"✅ 指纹 (前5位): {vec_base[0][:5]}")

    print("\n🔍 比较结果：")
    print("-" * 60)
    print(f"{'对比文本':<20} | {'相似度':<10} | {'向量指纹(前3位)'}")
    print("-" * 60)

    for text in compare_list:
        # 本地生成对比向量
        vec_target = model.encode([text])

        # 计算余弦相似度
        # model.encode 返回的是 2D array，所以取 [0]
        score = cosine_similarity(vec_base, vec_target)[0][0]

        fingerprint = vec_target[0][:3]
        print(f"{text:<20} | {score:.4f}     | {fingerprint}")


if __name__ == "__main__":
    local_embedding_test()
