import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. 加载环境变量 (最安全的做法)
load_dotenv()

def get_client():
    """封装客户端初始化逻辑，方便复用"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ 未找到 API Key，请检查 .env 文件")
    
    # 2. 初始化 V2 Client
    # http_options 参数在需要代理时非常有用，不需要代理可省略
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

def main():
    client = get_client()
    
    # 3. 发送请求
    # 使用 'gemini-2.0-flash-exp' 或 'gemini-1.5-flash' (目前最快)
    print("🤖 正在思考...")
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents="用 Python 程序员能懂的黑话，解释一下为什么在这个阶段要学习 AI？限 50 字以内。"
    )
    
    # 4. 打印结果
    print(f"💬 回复:\n{response.text}")

if __name__ == "__main__":
    main()