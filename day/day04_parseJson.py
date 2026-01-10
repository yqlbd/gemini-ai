from google import genai
import dotenv
import os
import re
import json
import asyncio

dotenv.load_dotenv


class ProjectAssistant:
    def __init__(self):
        # 🔑 核心升级：植入 CoT 思维链指令
        self.system_instruction = """
        你是一个精通项目管理的系统架构师。
        你的任务是根据用户的需求，拆解出详细的开发任务清单。

        【硬性约束】：
        1. 必须只返回 JSON 格式数据，严禁包含任何自然语言的解释或开场白。
        2. 必须严格遵守以下 JSON Schema（结构定义）：
        {
            "project_name": "项目名称",
            "total_modules": 模块总数,
            "details": [
                {
                    "module": "模块名称",
                    "priority": "High/Medium/Low",
                    "tasks": ["任务1描述", "任务2描述"]
                }
            ]
        }
        """

        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-3-pro-preview"  # 使用更强大的 Gemini 3 Pro 模型
        # 手动管理历史记录
        self.history = []

    async def ask(self, message):
        """发送消息并获取回复 (Async)"""
        print(f"⏳ 正在生成项目任务清单... (这是最耗时的步骤，请耐心等待)")

        # 构造请求：历史 + 当前问题
        messages = self.history + [{"role": "user", "parts": [{"text": message}]}]

        # 调用 API
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=messages,
            config={"system_instruction": self.system_instruction},
        )

        # 更新历史
        self.history.append({"role": "user", "parts": [{"text": message}]})
        self.history.append({"role": "model", "parts": [{"text": response.text}]})

        return response.text

    @staticmethod
    def clean_json_string(raw_text):
        # 1. 使用正则匹配 ```json 和 ``` 之间的内容
        # re.S 模式让 . 匹配换行符
        match = re.search(r"```json\s+(.*?)\s+```", raw_text, re.S)
        if match:
            clean_text = match.group(1)
        else:
            # 2. 如果没匹配到，尝试去掉可能存在的普通反引号
            clean_text = raw_text.strip().replace("```", "")
        return clean_text.strip()


async def main():
    # 实例化助手
    architect = ProjectAssistant()

    # 🎯 你的真实业务需求
    user_requirement = """
    我想做一个理财产品的秒杀方案。
    - 时间：每天9点准时开抢
    - 额度：100w
    - 预估并发：瞬间用户 50,000 人
    - 持续时间：预计 5s 内抢完
    - 核心红线：严禁超卖，这是金融产品。
    
    请帮我设计架构方案。
    """

    print(f"🚀 发送需求: {user_requirement.strip().splitlines()[0]}...")

    # 1. 异步调用 AI
    answer = await architect.ask(user_requirement)

    # 2. 打印部分结果到屏幕
    print("\n" + "=" * 40)
    print("🤖 架构师 Gemini 的回复 (预览前500字):")
    print("=" * 40)
    print(answer[:500] + "...\n(内容太长，已省略后续)")
    # 3. 清理并解析 JSON
    json.loads(architect.clean_json_string(answer))
    print("\n✅ 已成功解析为 JSON 格式！")


if __name__ == "__main__":
    asyncio.run(main())
