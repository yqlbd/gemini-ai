import os
import json
import requests
from dotenv import load_dotenv
import numpy as np

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
# 我们直接访问 REST API 的 HTTP 接口，绕过 SDK
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"


def get_embedding_raw(text):
    """
    不使用 google-genai 库，直接用 HTTP 协议发送请求。
    这是最底层的调用方式，没有任何中间商赚差价。
    """
    url = f"{BASE_URL}?key={API_KEY}"

    headers = {"Content-Type": "application/json"}

    # 构造原始 JSON 数据
    payload = {"content": {"parts": [{"text": text}]}}

    print(f"🚀 [HTTP直连] 发送: {text} ...")

    try:
        # 发送 POST 请求
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=10
        )

        if response.status_code != 200:
            print(f"❌ API 报错: {response.text}")
            return []

        # 解析 JSON
        data = response.json()
        embedding = data["embedding"]["values"]
        return embedding

    except Exception as e:
        print(f"❌ 网络错误: {e}")
        return []


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("🤖 初始化 HTTP 直连测试 (无 SDK 版)...")

    text_base = "胖墩墩"
    # 我们选两个之前“撞车”的冤家
    text_compare = "今天天气真好"

    # 1. 获取基准
    vec_base = get_embedding_raw(text_base)
    print(f"✅ 基准指纹 (前5位): {vec_base[:5]}")

    # 2. 获取对比
    vec_compare = get_embedding_raw(text_compare)
    print(f"✅ 对比指纹 (前5位): {vec_compare[:5]}")

    # 3. 最终审判
    if vec_base[:5] == vec_compare[:5]:
        print("\n😱 完蛋！HTTP 直连还是重复！这说明是 Google 账号/API 服务端的问题。")
    else:
        print("\n🎉 成功！指纹不一致！")
        score = cosine_similarity(vec_base, vec_compare)
        print(f"📊 真实相似度: {score:.4f} (预期应该很低)")
        print("💡 结论: 之前的 Bug 是 google-genai SDK 异步并发处理的问题。")


if __name__ == "__main__":
    # 如果没装 requests，请先 pip install requests
    main()
