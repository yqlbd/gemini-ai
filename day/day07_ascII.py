import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"


def get_embedding_safe(text):
    """
    使用 requests 的 json 参数，自动处理 Content-Type 和 UTF-8 编码，
    这是最不容易出编码问题的写法。
    """
    url = f"{BASE_URL}?key={API_KEY}"

    # 直接构造字典
    payload = {"content": {"parts": [{"text": text}]}}

    print(f"🚀 发送: {text}")

    # 注意：使用 json=payload，requests 会自动帮我们做 UTF-8 编码
    # 不要用 data=json.dumps(...)，那个容易出 Unicode 转义问题
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"❌ Error: {resp.text}")
            return None
        return resp.json()["embedding"]["values"]
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def main():
    print("🤖 正在进行 ASCII vs 中文 最终对比测试...\n")

    # 1. 测试纯英文 (这是破案的关键！)
    t1 = "Apple"
    t2 = "Banana"

    v1 = get_embedding_safe(t1)
    v2 = get_embedding_safe(t2)

    if v1 and v2:
        print(f"🍎 Apple 指纹前5位:  {v1[:5]}")
        print(f"🍌 Banana 指纹前5位: {v2[:5]}")

        if v1[:5] != v2[:5]:
            print("✅ 英文测试通过！API 是正常的！")
        else:
            print("😱 英文也重复？那才是真完蛋。")

    print("-" * 30)

    # 2. 测试中文 (使用 json=payload 修复后的)
    t3 = "胖墩墩"
    t4 = "今天天气真好"

    v3 = get_embedding_safe(t3)
    v4 = get_embedding_safe(t4)

    if v3 and v4:
        print(f"🐶 胖墩墩 指纹前5位: {v3[:5]}")
        print(f"☀️ 天气好 指纹前5位: {v4[:5]}")

        if v3[:5] != v4[:5]:
            print("🎉 中文也修复了！之前是 json 序列化编码的问题！")
        else:
            print("⚠️ 中文依然重复，可能需要检查系统 locale 设置。")


if __name__ == "__main__":
    main()
