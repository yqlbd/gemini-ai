import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"


def get_embedding_fixed(text):
    """
    【macOS 终极修复版】
    不依赖系统默认编码，强制使用 UTF-8 二进制流发送请求。
    """
    url = f"{BASE_URL}?key={API_KEY}"

    # 1. 显式声明我们发的是 UTF-8
    headers = {"Content-Type": "application/json; charset=utf-8"}

    # 2. 构造字典
    payload = {"content": {"parts": [{"text": text}]}}

    # 3. 【核心黑魔法】
    # 不要让 requests 帮我们序列化，我们自己动手！
    # ensure_ascii=False 保留中文原样
    # .encode('utf-8') 把它强制变成二进制流
    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    print(f"🚀 [UTF-8强转] 发送: {text}")

    try:
        # 注意：这里用 data=data_bytes，而不是 json=...
        response = requests.post(url, headers=headers, data=data_bytes, timeout=10)

        if response.status_code != 200:
            print(f"❌ API 报错: {response.text}")
            return []

        return response.json()["embedding"]["values"]

    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return []


def main():
    print("🤖 初始化 macOS 编码修复测试...\n")

    # 直接上高难度中文测试
    t1 = "胖墩墩"
    t2 = "今天天气真好"

    v1 = get_embedding_fixed(t1)
    v2 = get_embedding_fixed(t2)

    if v1 and v2:
        print(f"🐶 胖墩墩 指纹前5位: {v1[:5]}")
        print(f"☀️ 天气好 指纹前5位: {v2[:5]}")

        # 见证奇迹的时刻
        if v1[:5] != v2[:5]:
            print("\n🎉🎉🎉 成功了！中文指纹终于不一样了！")
            print(
                "💡 根本原因：macOS 环境下的 HTTP 请求头默认没带 charset=utf-8，导致中文丢失。"
            )
            print(
                "🚀 下一步：你可以放心地去搞 ChromaDB 了，把这个 get_embedding_fixed 函数带过去就行！"
            )
        else:
            print("\n😱 依然重复... 这简直不科学。")


if __name__ == "__main__":
    main()
