# app-T2puz82drGLj8AcJqLP0Z8d1
import requests
import json

# 1. 配置你的 Dify 地址和密钥
# 如果你还是本地 localhost:8080，地址通常是这个：
BASE_URL = "http://localhost/v1"
API_KEY = "app-T2puz82drGLj8AcJqLP0Z8d1"

# 2. 准备请求头
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# 3. 准备要问的问题
# userId 是必须的，Dify 用它来区分不同用户的记忆
data = {
    "inputs": {},  # 如果工作流里有定义的变量（比如开场白），填在这里
    "query": "胖墩墩会玩飞盘吗？",
    "response_mode": "blocking",  # blocking: 等全部生成完再返回; streaming: 流式打字机效果
    "conversation_id": "",  # 第一次留空，后面填具体的 ID 就能接上话
    "user": "yiqing_test_user",  # 你的用户标识
}

# 4. 发送请求
print("🤖 正在呼叫胖墩墩管家...")
try:
    response = requests.post(f"{BASE_URL}/chat-messages", headers=headers, json=data)
    response.raise_for_status()  # 检查有没有报错

    # 5. 解析结果
    result = response.json()
    answer = result.get("answer")

    print("-" * 30)
    print(f"🐶 回答: {answer}")
    print("-" * 30)

    # 打印一下消耗的 tokens (看看有没有走 RAG)
    metadata = result.get("metadata", {})
    print(f"📊 消耗: {metadata.get('usage', '未知')}")

except Exception as e:
    print(f"❌ 出错啦: {e}")
    # 如果出错，打印详细信息方便调试
    if "response" in locals():
        print(response.text)
