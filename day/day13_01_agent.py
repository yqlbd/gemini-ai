import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from utils.ai_tools import get_current_weather, calculate_dog_food, tools_list

# 1. 初始化
load_dotenv()

# 定义工具注册表 (这一步和昨天一样)
FUNCTION_MAP = {
    "get_current_weather": get_current_weather,
    "calculate_dog_food": calculate_dog_food,
}


class Agent:
    def __init__(self, model_id="gemini-2.0-flash"):
        """
        初始化 Agent，给它装上大脑 (Client) 和记忆 (History)
        """
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = model_id
        self.chat_history = []  # 记忆槽

        # 定义系统提示词：赋予它“思考”的人设
        self.system_instruction = """
        你是一个智能助手 (Agent)。
        你拥有查询天气和计算数据的工具。
        
        解决问题的步骤：
        1. 思考 (Thought): 分析用户的问题，决定需要什么信息。
        2. 行动 (Action): 如果需要外部信息，请调用相应的工具。
        3. 观察 (Observation): 查看工具返回的结果。
        4. 循环: 如果信息不足，重复上述步骤。
        5. 回答 (Answer): 当信息充足时，直接回答用户。
        
        注意：禁止编造数据，必须依赖工具返回的结果。
        """

    def chat(self, user_query):
        """
        Agent 的主循环：思考 -> 决策 -> 执行 -> 再思考
        """
        print(f"\n🟢 [用户]: {user_query}")

        # 1. 把用户的问题加入记忆
        self.chat_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
        )

        # 2. 开启“思考循环” (Max Turns 防止死循环)
        max_turns = 5
        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1
            print(f"🔄 [第 {turn_count} 轮思考]...")

            # --- A. 调用 LLM 大脑 ---
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=self.chat_history,
                config=types.GenerateContentConfig(
                    tools=tools_list,
                    temperature=0.0,
                    system_instruction=self.system_instruction,
                    automatic_function_calling={"disable": True},  # 关键：我们手动控制
                ),
            )

            # --- B. 检查 LLM 想干什么 ---
            # ✅ 修复方案：先判断是否包含工具调用 (Function Call)
            # 如果有工具调用，直接进入处理流程，不要去读 response.text
            if self._has_function_call(response):
                # 先把“我想调工具”这个念头存入历史
                self.chat_history.append(response.candidates[0].content)

                # 执行所有请求的工具
                self._execute_tool_calls(response.candidates[0].content.parts)
                continue  # 直接进入下一轮循环，完美避开警告

            # ✅ 情况 2: 如果没有工具调用，才去检查有没有文本
            # 此时访问 response.text 是安全的，因为我们已经排除了纯 FunctionCall 的情况
            if response.text:
                print(f"🤖 [Agent 回答]: {response.text}")
                self.chat_history.append(response.candidates[0].content)
                return response.text

            # 情况 3: 异常处理
            print("⚠️ [Agent] 似乎卡住了...")
            break

    def _has_function_call(self, response):
        """辅助函数：检查响应里有没有工具调用请求"""
        if not response.candidates:
            return False
        for part in response.candidates[0].content.parts:
            if part.function_call:
                return True
        return False

    def _execute_tool_calls(self, parts):
        """
        [修复版] 执行工具调用，并将结果一次性写回历史记录
        """
        # 1. 准备一个列表，用来收集所有的工具执行结果
        response_parts = []

        for part in parts:
            if part.function_call:
                fn_name = part.function_call.name
                fn_args = part.function_call.args

                print(f"🔨 [执行方法] {fn_name} | 参数: {fn_args}")

                # 动态分发
                if fn_name in FUNCTION_MAP:
                    try:
                        result = FUNCTION_MAP[fn_name](**fn_args)
                        print(f"📦 [方法结果] {result}")
                    except Exception as e:
                        result = f"Error: {str(e)}"
                else:
                    result = f"Error: Unknown tool {fn_name}"

                # 2. 构造 Part 对象，但先不存入 history，而是加入列表
                response_part = types.Part.from_function_response(
                    name=fn_name, response={"result": result}
                )
                response_parts.append(response_part)

        # 3. 循环结束后，把所有结果打包成【一条】Content 存入历史
        if response_parts:
            self.chat_history.append(types.Content(role="tool", parts=response_parts))


# --- 🚀 测试代码 ---
if __name__ == "__main__":
    # 实例化一个 Agent
    my_agent = Agent()

    # 挑战：一个需要【两步】才能解决的问题
    # 1. 查天气 -> 2. (AI 可能会判断) -> 3. 回答
    # 比如我们问：
    # "胖墩墩在常州，今天天气适合去公园吗？如果去的话，它 8.5kg 跑完回来该吃多少？"
    # 这个问题迫使 AI 必须调用两个工具，并且把逻辑串起来。

    query = "胖墩墩现在在常州，8.5kg。请帮我判断今天适不适合带它去公园，以及如果运动量大，它今天该吃多少狗粮？"
    my_agent.chat(query)
