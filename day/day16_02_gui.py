import streamlit as st
import requests
import json
import uuid

# --- 配置区 ---
BASE_URL = "http://localhost/v1"
API_KEY = "app-T2puz82drGLj8AcJqLP0Z8d1"

# --- 页面设置 ---
st.set_page_config(page_title="胖墩墩专属管家", page_icon="🐶")
st.title("🐶 胖墩墩专属管家")
st.caption("Powered by Dify + Gemini")

# --- 初始化会话状态 ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # 生成一个唯一ID，让Dify记住上下文
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 显示历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 处理用户输入 ---
if prompt := st.chat_input("问问关于胖墩墩的事..."):
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 呼叫 Dify API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 思考中...")

        try:
            # 构造请求
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "conversation_id": st.session_state.session_id,  # 传入会话ID，实现多轮对话
                "user": "yiqing_streamlit_user",
            }

            # 发送请求
            response = requests.post(
                f"{BASE_URL}/chat-messages", headers=headers, json=payload
            )
            response.raise_for_status()

            # 解析结果
            result = response.json()
            answer = result.get("answer", "我好像卡住了...")

            # 显示回答
            message_placeholder.markdown(answer)

            # 存入历史
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            message_placeholder.error(f"出错啦: {e}")
