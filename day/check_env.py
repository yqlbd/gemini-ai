import os
import sys

def check_env():
    print(f"--- 一清的 AI 开发环境检查 ---")
    # 1. 检查 Python 路径（确保不是系统自带的 /usr/bin/python）
    print(f"🐍 Python 路径: {sys.executable}")
    
    # 2. 检查虚拟环境
    is_venv = sys.prefix != sys.base_prefix
    print(f"📦 虚拟环境激活: {'✅' if is_venv else '❌ (请务必激活 venv！)'}")
    
    # 3. 检查环境变量
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"🔑 API Key 配置: {'✅' if api_key else '❌ (环境变量未找到)'}")
    
    # 4. 检查 SDK
    try:
        from google import genai
        print(f"🚀 Google GenAI SDK: ✅ 已安装")
    except ImportError:
        print(f"🚀 Google GenAI SDK: ❌ 未找到 (pip install google-genai)")

if __name__ == "__main__":
    check_env()