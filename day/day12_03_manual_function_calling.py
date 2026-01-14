"""
LLM自动调用function calling
"""

from google import genai
from google.genai import types
import os
import asyncio


# 自定义的api，模拟获取天气数据
def get_weather_info(city: str = "上海") -> str:
    mock_data = {"上海": "18度，晴转多云", "北京": "20度，晴", "广州": "16度，阴转小雨"}
    return mock_data.get(city, "未找到当地天气，请联系气象部门")


def get_name_info(role: str = "小狗") -> str:
    mock_data = {"妈妈": "霍妮媛", "小狗": "胖墩墩", "我": "赵一清"}
    return mock_data.get(role, "未找到对应信息")


model_id = "gemini-2.0-flash"

tool_list = [get_weather_info, get_name_info]
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
FUNCTION_MAP = {"get_weather_info": get_weather_info, "get_name_info": get_name_info}


# 调用
async def get_gemini_response(question: str) -> None:
    chat_history = []
    if question:
        chat_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=question)])
        )
        print(f"👨的提问：{question}")
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction="你是个AI助理，如果询问名字，请调用fucntion_calling",
                temperature=0.3,
                tools=tool_list,
                # 增加这个配置，用来手动调用
                automatic_function_calling={"disable": True},
            ),
        )
        candidates = response.candidates[0]
        # 防御性编程：有时候 Gemini 可能会返回空内容
        if not candidates.content or not candidates.content.parts:
            print("❌ 错误：Gemini 返回了空内容")
            return

        for part in candidates.content.parts:
            # 如果是function_calling
            if part.function_call:
                fn_name = part.function_call.name
                fn_args = part.function_call.args
                print(f"🤖 [动态分发] 调用方法: {fn_name} | 参数: {fn_args}")
                if fn_name in FUNCTION_MAP:
                    target_function = FUNCTION_MAP[fn_name]
                    try:
                        # ⭐ 魔法时刻：**fn_args 是 Python 的参数解包
                        # 只要 Gemini 给的参数名(key)和函数定义的参数名一致，它就会自动填入
                        tool_result = target_function(**fn_args)
                        print(f"📦 [执行成功] 结果: {tool_result}")
                    except Exception as e:
                        tool_result = f"函数执行出错: {str(e)}"
                        print(f"❌ [执行失败] {str(e)}")
                else:
                    tool_result = f"错误：未知的工具 {fn_name}"
                    print(f"⚠️ [未知工具] {fn_name}")

                function_response_part = types.Part.from_function_response(
                    name=fn_name, response={"result": tool_result}
                )

                # 更新历史并发送
                # 拼装model的functioncall信息
                chat_history.append(candidates.content)
                # 拼装tool返回的信息
                chat_history.append(
                    types.Content(role="tool", parts=[function_response_part])
                )
                # 最后的数据示例：[{'role':'user',[{'text':'广州天气如何',...}]},{'role':''model',[{'funtion call':...}]},{'role':'tool',...}]
                final_response = await client.aio.models.generate_content(
                    model=model_id,
                    contents=chat_history,
                    config=types.GenerateContentConfig(temperature=2.0),
                )
                print(f"🤖的回复: {final_response.text}")
                return

        if response.text:
            print(f"🤖的回复：{response.text}")


async def main():
    # 所有的业务逻辑都放在这个 main 异步函数里
    question = "你是什么模型？"
    await get_gemini_response(question=question)

    question2 = "广州天气如何？"
    await get_gemini_response(question=question2)

    question3 = "小狗叫啥名字？"
    await get_gemini_response(question=question3)


if __name__ == "__main__":
    asyncio.run(main())
