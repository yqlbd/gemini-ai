"""
LLM自动调用function calling
"""

from google import genai
from google.genai import types
import os
import asyncio


# 自定义的api，模拟获取天气数据
def get_weather_info(city: str = "上海") -> str:
    print(f"======手搓api接受到调用，参数是{city}")
    mock_data = {"上海": "18度，晴转多云", "北京": "20度，晴", "广州": "16度，阴转小雨"}
    return mock_data.get(city, "未找到当地天气，请联系气象部门")


tool_list = [get_weather_info]
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# 调用
async def get_gemini_response(question: str) -> None:
    if question:
        print(f"👨的提问：{question}")
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=question,
            config=types.GenerateContentConfig(
                # system_instruction="",
                temperature=0.3,
                tools=tool_list,
            ),
        )

        if response.text:
            print(f"🤖的回复：{response.text}")


async def main():
    # 所有的业务逻辑都放在这个 main 异步函数里
    question = "你是什么模型？"
    await get_gemini_response(question=question)

    question2 = "广州天气如何？"
    await get_gemini_response(question=question2)


if __name__ == "__main__":
    asyncio.run(main())
