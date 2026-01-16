import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

from utils.ai_tools import (
    tools_list,
    get_current_weather,
    calculate_dog_food,
    search_knowledge_base,
)

load_dotenv()

# 🔥 1. 更新函数注册表 (Function Registry)
FUNCTION_MAP = {
    "get_current_weather": get_current_weather,
    "calculate_dog_food": calculate_dog_food,
    "search_knowledge_base": search_knowledge_base,  # 👈 新增：RAG 工具注册
}


class AdvancedAgent:
    def __init__(self, model_id="gemini-2.0-flash"):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = model_id
        self.chat_history = []

        # 🔥 2. 升级系统提示词 (注入灵魂)
        self.system_instruction = """
        你是一个全能型智能助手，名字叫“胖墩墩管家”。
        你拥有以下强力工具：
        1. `search_knowledge_base`: **核心工具**。当问题涉及“我”、“胖墩墩”、“日记”、“以前”或“笔记”等私有信息时，**必须优先调用**此工具查库，不要瞎编。
        2. `get_current_weather`: 查询实时天气。
        3. `calculate_dog_food`: 计算狗粮用量。

        思考与行动策略 (ReAct Loop):
        - 收到问题后，先分析需要哪些信息。
        - 如果是私有知识（如“我上周干了啥”），请调 `search_knowledge_base`。
        - 如果是客观事实（如“常州天气”），请调 `get_current_weather`。
        - 拿到工具结果后，结合你的常识进行综合回答。
        """

    def chat(self, user_query):
        print(f"\n🟢 [用户]: {user_query}")
        self.chat_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
        )

        max_turns = 5
        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1
            print(f"🔄 [第 {turn_count} 轮思考]...")

            # 调用 LLM
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=self.chat_history,
                config=types.GenerateContentConfig(
                    tools=tools_list,
                    temperature=0.0,
                    system_instruction=self.system_instruction,
                    automatic_function_calling={"disable": True},  # 坚持手动挡
                ),
            )

            # 检查是否有工具调用
            if self._has_function_call(response):
                self.chat_history.append(response.candidates[0].content)
                self._execute_tool_calls(response.candidates[0].content.parts)
                continue  # 继续循环，让 AI 看到结果

            # 如果没有工具调用，且有文本，说明思考结束
            if response.text:
                print(f"🤖 [最终回答]: {response.text}")
                self.chat_history.append(response.candidates[0].content)
                return response.text

            print("⚠️ [Agent] 无输出，跳出循环")
            break

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

                print(f"🔨 [调用工具] {fn_name} | 参数: {fn_args}")

                if fn_name in FUNCTION_MAP:
                    try:
                        # 动态执行函数
                        result = FUNCTION_MAP[fn_name](**fn_args)
                        # 为了终端显示好看，如果是长文本(RAG结果)，截断显示
                        display_result = (
                            str(result)[:100] + "..."
                            if len(str(result)) > 100
                            else result
                        )
                        print(f"📦 [返回结果] {display_result}")
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


if __name__ == "__main__":
    agent = AdvancedAgent()

    # 🧪 测试用例：RAG + Function Calling 的混合双打
    # 这个问题需要：
    # 1. 查 RAG -> 知道胖墩墩是"容易感冒"或者"喜欢玩飞盘" (假设日记里有)
    # 2. 查 Weather -> 知道常州今天下雨
    # 3. 综合建议

    query = "结合胖墩墩的身体情况（查日记），看看今天常州的天气适合带它去户外玩吗？"
    agent.chat(query)
