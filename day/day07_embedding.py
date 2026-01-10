import os
import asyncio
from google import genai
from dotenv import load_dotenv
import numpy as np

# 1. 加载环境
load_dotenv()


class EmbeddingMaster:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        # 使用专门的 embedding 模型，text-embedding-004 是目前最新且性价比最高的
        self.model_id = "text-embedding-004"

    async def get_embedding(self, text):
        """调用 API 将文本转换为向量"""
        try:
            result = await self.client.aio.models.embed_content(
                model=self.model_id, contents=text
            )
            # Embedding 通常在 result.embeddings[0].values 中 (视 SDK 版本而定，V2 SDK 结构如下)
            # 或者 result.embedding.values
            return result.embeddings[0].values
        except Exception as e:
            print(f"❌ Embedding 生成失败: {e}")
            return []

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """
        计算余弦相似度 (Cosine Similarity)
        公式: (A . B) / (||A|| * ||B||)
        范围: -1 到 1，越接近 1 表示越相似
        """
        if not vec_a or not vec_b:
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def main():
    master = EmbeddingMaster()

    print("🤖 正在初始化 Embedding 测试...")

    # 测试数据：我们要看看计算机是否认为 "胖墩墩" 和 "小狗" 是相似的
    text_base = "胖墩墩"

    compare_list = [
        "一只可爱的小狗",
        "边境牧羊犬",  # 语义强相关（它是这个品种）
        "美味的红烧肉",  # 语义弱相关（都是名词，但类别不同）
        "Python编程语言",  # 语义无关
        "今天天气真好",  # 语义无关
    ]

    # 1. 获取基准向量
    vec_base = await master.get_embedding(text_base)
    print(f"✅ '{text_base}' 向量化完成，维度: {len(vec_base)} (前5位: {vec_base[:5]})")

    for text in compare_list:
        vec_target = await master.get_embedding(text)
        print(
            f"✅ '{text}' 向量化完成，维度: {len(vec_target)} (前5位: {vec_target[:5]})"
        )

    print("\n🔍 开始语义距离计算：")
    print("-" * 40)
    print(f"{'文本':<20} | {'相似度 (Score)':<10} | {'评价'}")
    print("-" * 40)

    for text in compare_list:

        # 2. 获取对比向量
        vec_target = await master.get_embedding(text)

        # 3. 计算相似度
        score = master.cosine_similarity(vec_base, vec_target)

        # 简单的评价逻辑
        tag = "⭐⭐⭐" if score > 0.6 else ("⭐" if score > 0.3 else "❌")

        print(f"{text:<20} | {score:.4f}       | {tag}")


if __name__ == "__main__":
    asyncio.run(main())
