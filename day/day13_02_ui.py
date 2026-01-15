# 文件路径: day/day13_ui.py

import streamlit as st
import sys
import os

# 确保能找到 day 目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from day.day13_01_agent import Agent  # 👈 直接复用你刚才写好的 Agent 类

# --- 页面配置 ---
st.set_page_config(page_title="胖墩墩智能助理 (Agent版)", page_icon="🐶")
st.title("🐶 胖墩墩智能助理 (Agent版)")
st.caption("🚀 Powered by Gemini 2.0 + Function Calling + ReAct Loop")

# --- 初始化 Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# 初始化 Agent (利用缓存，避免每次重跑都重新连接)
@st.cache_resource
def get_agent():
    return Agent()


agent = get_agent()

# --- 侧边栏 ---
with st.sidebar:
    st.header("🛠️ 调试面板")
    st.info(
        "这个版本集成了 Function Calling。\n你可以问：\n\n1. 胖墩墩现在 8.5kg 活泼，该吃多少？\n2. 常州今天天气适合遛狗吗？"
    )
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# --- 展示历史消息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 处理用户输入 ---
if prompt := st.chat_input("问点什么吧..."):
    # 1. 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Agent 思考与回复
    with st.chat_message("assistant"):
        # 创建一个状态容器，用来显示 Agent 的思考过程
        status_container = st.status("🧠 Agent 正在思考...", expanded=True)

        # ⚠️ 关键技巧：为了让 Agent 的 print 输出显示在网页上，我们需要“劫持” stdout
        # 这里为了演示简单，我们先只显示最终结果。
        # 如果要显示中间步骤，需要改造 Agent 类支持 callback (那是进阶课内容)

        try:
            # 这里的 agent.chat 目前是 print 到终端的
            # 我们稍微改一下 day13_agent.py 让它 return 结果
            # 或者直接调用，看终端日志，网页显示最终结果
            response_text = agent.chat(prompt)

            status_container.update(
                label="✅ 思考完成！", state="complete", expanded=False
            )

            st.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )

        except Exception as e:
            status_container.update(label="❌ 出错了", state="error")
            st.error(f"Agent 运行出错: {str(e)}")
