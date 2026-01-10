import os
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict
from google import genai
from dotenv import load_dotenv
import numpy as np

# 1. 配置日志：我们要看清楚每一次请求到底发了什么
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EmbeddingService")

load_dotenv()


@dataclass
class TextVector:
    """定义一个数据结构，把文本和它的向量绑定在一起，防止乱套"""

    text: str
    vector: List[float] = None
    fingerprint: list = None  # 存前5位指纹用于调试


class CloudEmbeddingService:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ 未找到 GEMINI_API_KEY，请检查 .env 文件")

        self.client = genai.Client(api_key=api_key)
        self.model_id = "text-embedding-004"

    async def fetch_single_embedding(self, text: str) -> TextVector:
        """
        处理单个文本的向量化。
        这个函数会被并发调用，每个调用都是独立的，绝对不会搞混变量。
        """
        try:
            # 🔍 照妖镜：打印当前正在请求的文本
            logger.info(f"🚀 发送请求: '{text}'")

            result = await self.client.aio.models.embed_content(
                model=self.model_id, contents=text
            )

            vec = result.embeddings[0].values

            # 🔍 结果校验：打印返回向量的前5位
            fingerprint = vec[:5]
            logger.info(f"✅ 收到响应: '{text}' -> 指纹: {fingerprint}")

            return TextVector(text=text, vector=vec, fingerprint=fingerprint)

        except Exception as e:
            logger.error(f"❌ 失败: '{text}' - 原因: {e}")
            return TextVector(text=text, vector=[])

    async def fetch_batch(self, texts: List[str]) -> Dict[str, TextVector]:
        """核心重构：并发获取所有向量"""
        tasks = [self.fetch_single_embedding(t) for t in texts]

        # 并发执行所有任务！
        logger.info(f"⚡ 开始并发处理 {len(texts)} 个任务...")
        results = await asyncio.gather(*tasks)

        # 将结果转为字典方便查询 { "文本": TextVector对象 }
        return {res.text: res for res in results}


class VectorMath:
    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        if not vec_a or not vec_b:
            return 0.0
        a = np.array(vec_a)
        b = np.array(vec_b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


async def main():
    service = CloudEmbeddingService()

    # 1. 准备数据
    text_base = "胖墩墩"
    compare_list = [
        "胖墩墩",  # 自己
        "一只可爱的小狗",  # 强相关
        "今天天气真好",  # 无关
        "Python编程语言",  # 无关
    ]

    # 去重并合并，一次性查完
    all_texts = list(set([text_base] + compare_list))

    # 2. 并发获取（这是解决问题的关键步骤）
    # 所有的请求同时发出去，这就避免了循环变量复用的问题
    vector_map = await service.fetch_batch(all_texts)

    # 3. 验证指纹（显微镜环节）
    print("\n🕵️‍♂️ 指纹一致性检查 (如果不通过，说明 API 有问题):")
    base_obj = vector_map[text_base]
    print(f"📌 基准 [{text_base}] 指纹: {base_obj.fingerprint}")

    # 4. 计算并展示
    print(f"\n📊 '{text_base}' 的相似度报告:")
    print("-" * 60)
    print(f"{'对比文本':<20} | {'相似度':<10} | {'指纹对比结果'}")
    print("-" * 60)

    for text in compare_list:
        target_obj = vector_map[text]
        score = VectorMath.cosine_similarity(base_obj.vector, target_obj.vector)

        # 自动判断指纹是否重复
        is_same_fingerprint = base_obj.fingerprint == target_obj.fingerprint
        status = (
            "⚠️ 指纹完全重合(异常)"
            if is_same_fingerprint and text != text_base
            else "✅ 正常"
        )
        if text == text_base:
            status = "✅ 自身(正常)"

        print(f"{text:<20} | {score:.4f}     | {status}")


if __name__ == "__main__":
    asyncio.run(main())
