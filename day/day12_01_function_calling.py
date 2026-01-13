from dotenv import load_dotenv

# 1.将环境变量加载到上下文
load_dotenv()

from google import genai
import os

# 2.获取gemini client，告诉gemini我们有那些方法可以调用
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
model_id = "gemini-2.0-flash-exp"
# 这是手动维护的多轮对话列表，自己维护
chat_history = []

from google.genai import types
from utils.ai_tools import get_current_weather, calculate_dog_food

tools_list = [get_current_weather, calculate_dog_food]


def chat_with_tools(user_query):
    print(f"\n🧑 用户提问: {user_query}")
    chat_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    )

    # --- 第 1 轮交互：用户提问 -> Gemini 思考 ---
    # Gemini 会判断：是直接回答，还是申请调用工具？
    response = client.models.generate_content(
        model=model_id,
        contents=chat_history,
        config=types.GenerateContentConfig(
            tools=tools_list,  # 🔥 关键：把工具包挂载上去
            temperature=0.0,  # 工具调用建议设为 0，让它逻辑更严谨
            system_instruction="你是一个助手。如果用户问天气或计算狗粮，**必须**调用工具获取数据，禁止直接回答或编造。",
            automatic_function_calling={"disable": True},
        ),
    )
    # 检查gemini的反应
    # response.candidates[0].content.parts 里面可能有文本，也可能有 function_call
    parts = response.candidates[0].content.parts
    # --- 第 2 轮交互：处理工具调用 (如果有) ---
    for part in parts:
        # 如果 Gemini 发出了“函数调用请求”
        if part.function_call:
            fn_name = part.function_call.name
            fn_args = part.function_call.args

            print(f"🤖 Gemini 请求调用工具: 【{fn_name}】 参数: {fn_args}")

            # === 这里就是“Python 老板”介入的地方 ===
            # 我们手动执行对应的函数
            tool_result = None
            if fn_name == "get_current_weather":
                tool_result = get_current_weather(city=fn_args["city"])
            elif fn_name == "calculate_dog_food":
                # 注意：Gemini 有时传回来的是浮点数，稍微做下类型转换
                tool_result = calculate_dog_food(
                    weight_kg=fn_args.get("weight_kg"),
                    is_active=fn_args.get("is_active", True),
                )

            print(f"📦 工具运行结果: {tool_result}")

            # === 把结果“喂”回给 Gemini ===
            # 我们需要构造一个特殊的响应，告诉 AI：“你刚才要跑的函数，结果在这里”
            # 这里的格式是固定的，必须包含 id, name, response
            function_response_part = types.Part.from_function_response(
                name=fn_name, response={"result": tool_result}
            )

            # 把“Gemini 的请求”和“我们的运行结果”都塞进历史记录
            # 这样 AI 才知道前因后果
            chat_history.append(response.candidates[0].content)  # 存入它刚才的请求
            chat_history.append(
                types.Content(role="tool", parts=[function_response_part])
            )

            # --- 第 3 轮交互：拿到结果 -> Gemini 生成最终人话 ---
            final_response = client.models.generate_content(
                model=model_id,
                contents=chat_history,
                # 这轮不需要tools了，或者带着也行，通常生成最终回答不需要再调用工具
            )
            print(f"🤖 Gemini 最终回答: {final_response.text}")
            return final_response.text

    # 如果没有调用工具（比如用户只是问好），直接打印文本
    if response.text:
        print(f"🤖 Gemini 直接回答: {response.text}")
        return response.text


if __name__ == "__main__":
    # 测试 1: 简单的天气查询
    chat_with_tools("常州天气如何？")

    # print("-" * 50)

    # 测试 2: 复杂的计算 (结合了你的个人信息：胖墩墩 8.5kg)
    # 注意：我在 Prompt 里没说体重，看 AI 会不会幻觉（它应该会问体重，或者瞎猜）
    # 为了测试效果，我们直接在问题里带上体重
    # chat_with_tools("胖墩墩现在 8.5kg 了，而且特皮，每天该吃多少狗粮？")
