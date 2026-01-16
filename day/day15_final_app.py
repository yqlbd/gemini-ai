# 文件路径: day/day15_final_app.py

import streamlit as st
import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 确保能引用到 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from day.utils.ai_tools import (
    tools_list,
    get_current_weather,
    calculate_dog_food,
    search_knowledge_base,
)

# 1. 初始化环境
load_dotenv()
st.set_page_config(page_title="胖墩墩全能管家", page_icon="🐶", layout="centered")

# --- 核心：定义支持 UI 反馈的 Agent ---
# 我们把 Day 14 的逻辑搬过来，并加上 Streamlit 的视觉反馈
FUNCTION_MAP = {
    "get_current_weather": get_current_weather,
    "calculate_dog_food": calculate_dog_food,
    "search_knowledge_base": search_knowledge_base,
}


class StreamlitAgent:
    def __init__(self, model_id="gemini-2.0-flash-exp"):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = model_id
        self.chat_history = []

        # Day 14 的终极 Prompt
        self.system_instruction = """
        你是一个全能型智能助手，名字叫“胖墩墩管家”。
        你拥有以下强力工具：
        1. `search_knowledge_base`: **核心工具**。当问题涉及“我”、“胖墩墩”、“日记”、“以前”或“笔记”等私有信息时，**必须优先调用**此工具查库。
        2. `get_current_weather`: 查询实时天气。
        3. `calculate_dog_food`: 计算狗粮用量。

        思考与行动策略 (ReAct Loop):
        - 收到问题后，先分析需要哪些信息。
        - 遇到私有知识，调 `search_knowledge_base`。
        - 遇到客观事实，调 `get_current_weather`。
        - 拿到工具结果后，结合你的常识进行综合回答。
        """

    def chat(self, user_query):
        # 把用户问题加入历史
        self.chat_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
        )

        max_turns = 5
        turn_count = 0

        # 在 UI 上显示一个状态容器
        with st.status("🧠 大脑飞速运转中...", expanded=True) as status:

            while turn_count < max_turns:
                turn_count += 1
                st.write(f"🔄 第 {turn_count} 轮思考...")

                # 调用 Gemini
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=self.chat_history,
                    config=types.GenerateContentConfig(
                        tools=tools_list,
                        temperature=0.0,
                        system_instruction=self.system_instruction,
                        automatic_function_calling={"disable": True},
                    ),
                )

                # 1. 检查工具调用
                if self._has_function_call(response):
                    # 在 UI 上告知用户正在调用工具
                    call_names = [
                        part.function_call.name
                        for part in response.candidates[0].content.parts
                        if part.function_call
                    ]
                    st.info(f"🛠️ 正在调用工具: {', '.join(call_names)}")

                    self.chat_history.append(response.candidates[0].content)
                    self._execute_tool_calls(response.candidates[0].content.parts)
                    continue

                # 2. 检查文本回答
                if response.text:
                    status.update(
                        label="✅ 思考完成！", state="complete", expanded=False
                    )
                    self.chat_history.append(response.candidates[0].content)
                    return response.text

                break

            status.update(label="⚠️ 思考超时或中断", state="error")
            return "抱歉，我思考太久了，有点乱..."

    def _has_function_call(self, response):
        if not response.candidates:
            return False
        for part in response.candidates[0].content.parts:
            if part.function_call:
                return True
        return False

    def _execute_tool_calls(self, parts):
        response_parts = []
        for part in parts:
            if part.function_call:
                fn_name = part.function_call.name
                fn_args = part.function_call.args

                # UI 反馈：如果是查库，显示个特别的 Toast
                if fn_name == "search_knowledge_base":
                    st.toast(f"📚 正在翻阅日记库...", icon="📖")
                elif fn_name == "get_current_weather":
                    st.toast(f"☁️ 正在查询天气...", icon="🌦️")

                if fn_name in FUNCTION_MAP:
                    try:
                        result = FUNCTION_MAP[fn_name](**fn_args)
                        # 在状态栏里折叠显示详细结果，避免刷屏
                        with st.expander(f"📦 工具 {fn_name} 返回结果"):
                            st.code(str(result)[:500])  # 只显示前500字
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Error: Unknown tool {fn_name}"

                response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name, response={"result": result}
                    )
                )

        if response_parts:
            self.chat_history.append(types.Content(role="tool", parts=response_parts))


# --- UI 主程序逻辑 ---

st.title("🐶 胖墩墩全能管家")
st.caption("🚀 架构: Streamlit + Gemini 2.0 + RAG Memory + Tools")

# 侧边栏
with st.sidebar:
    st.image(
        "img/DSC01879.jpeg", caption="我是胖墩墩", use_container_width=True
    )  # 假设你上传了这张图)
    st.header("功能展示")
    st.markdown(
        """
    试着问我：
    1. **查天气**: 常州今天天气咋样？
    2. **查算术**: 8.5kg 狗吃多少？
    3. **查记忆 (RAG)**: 胖墩墩以前玩过飞盘吗？
    4. **混合双打**: 
       > *结合胖墩墩的身体情况（查日记），看看今天常州的天气适合带它去户外玩吗？*
    """
    )
    if st.button("🗑️ 清空对话记忆"):
        st.session_state.messages = []
        st.session_state.agent = StreamlitAgent()  # 重置 Agent
        st.rerun()

# 初始化 Session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = StreamlitAgent()

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理输入
if prompt := st.chat_input("召唤全能管家..."):
    # 1. 显示用户输入
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Agent 开始表演
    with st.chat_message("assistant"):
        # 直接调用 Agent，内部会处理 UI 状态显示
        response_text = st.session_state.agent.chat(prompt)

        # 显示最终结果
        st.markdown(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
