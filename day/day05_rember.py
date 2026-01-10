from utils.ai_tools import AIToolkit
import json
from day04_parseJson_final import ProjectAssistant
import asyncio
from google import genai
import os


class RebuildAssiantant:
    def __init__(self):
        self.system_instruction = """
        ### 角色：
        你是一个资深的重构工程师，善于将伪代码按照要求进行优化和重构，达到生产的标准。

        你的任务：
        1. 根据伪代码和安全审计架构师的反馈，进行代码重构。
        2. 输出JSON格式的数据，严禁包含任何自然语言的解释或开场白。
        
        输出示例：
        {
            "repseudocode": "local key = KEYS[1] local stock = redis.call('HGET', key, 'stock') if stock == nil or tonumber(stock) <= 0 then return 0 end local newStock = tonumber(stock) - 1 if newStock < 0 then return 0 end redis.call('HSET', key, 'stock', newStock) return 1"
        }
        """
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-3-pro-preview"

    async def ask(self, message):
        print(f"⏳ 正在由重构工程师处理... (这是非常耗时的步骤，请耐心等待)")
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=[{"role": "user", "parts": [{"text": message}]}],
            config={"system_instruction": self.system_instruction},
        )
        return response.text


class ReviewAssistant:
    def __init__(self):
        self.system_instruction = """
        ### 角色：
        你现在是一个专门寻找 12306 级别并发漏洞的首席安全架构师。请对 Developer 提交的 Redis Lua 脚本进行‘地狱级’审计。如果发现没有使用 {} Hash Tag 或者没有处理原子性溢出，请直接给出 0 分并要求重构！

        你的任务：
        1. 审查 Developer 提交的伪代码。
        2. 重点寻找：竞态条件（Race Condition）、超卖风险、死锁、性能瓶颈、幂等性缺失。
        3. 必须以 JSON 格式输出，包含 "score" (0-10分) 和 "critiques" (建议列表)

        JSON 结构示例：
        {
            "score": 5,
            "critiques": ["Lua脚本未处理库存为负数的情况", "缺少分布式锁的过期保护"]
        }
        """
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-3-pro-preview"

    async def ask(self, message):
        print(f"⏳ 正在给安全架构师审计... (这是非常耗时的步骤，请耐心等待)")
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=[{"role": "user", "parts": [{"text": message}]}],
            config={"system_instruction": self.system_instruction},
        )
        return response.text


async def main():
    # 1. 加载昨天的“记忆”
    history_data, file_name = AIToolkit.load_latest_json("project_tasks_db")

    if history_data:
        print(f"✅ 已加载历史记忆: {file_name}")

        # 2. 构造“带记忆”的提问
        str1 = f"""
        我们之前的设计方案如下：
        {json.dumps(history_data, ensure_ascii=False)}
        """
        str2 = """
        请基于上述架构，为其中的“分布式缓存扣减核心”模块编写一段伪代码实现。

        ### 格式要求：
        1. 必须只返回 JSON 格式数据，严禁包含任何自然语言的解释或开场白。
        2. 必须严格遵守以下 JSON Schema（结构定义）：
        {
            "type": "object",
            "properties": {
                "module": { "type": "string", "description": "模块名称" },
                "pseudocode": { "type": "string", "description": "伪代码实现" }
            },
            "required": ["module", "pseudocode"]
        }
        """
        context_prompt = f"{str1}\n{str2}"
        assistant = ProjectAssistant()

        # 3. 交给 AI 继续工作
        answer = await assistant.ask(context_prompt)
        print("\n" + "=" * 40)
        print("🤖 系统架构师 的回复")
        print("=" * 40)
        # print(answer)
        final_dict = json.loads(AIToolkit.clean_json_string_2(answer), strict=False)
        print("\n✅ 已成功解析为 JSON 格式！")
        print(final_dict)
        # 4.交给审计AI检查
        review_assistant = ReviewAssistant()
        review_answer = await review_assistant.ask(final_dict["pseudocode"])
        print("\n" + "=" * 40)
        print("🤖 安全架构师 的回复")
        print("=" * 40)
        print(review_answer)

        review_answer_dict = json.loads(
            AIToolkit.clean_json_string_2(review_answer), strict=False
        )

        if review_answer_dict["score"] < 6:
            print("\n❌ 安全审计未通过，分数低于 6 分，要求重构！\n")
            # 交给重构AI
            rebuild_assistant = RebuildAssiantant()
            rebulid_message = f"""
            之前的代码如下：{final_dict['pseudocode']}
            安全审计师的反馈如下：{review_answer_dict['critiques']}
            请根据反馈进行重构。
            """
            rebuild_answer = await rebuild_assistant.ask(rebulid_message)
            print("\n" + "=" * 40)
            print("🤖 重构工程师 的回复")
            print("=" * 40)
            print(rebuild_answer)
            rebuild_answer_dict = json.loads(
                AIToolkit.clean_json_string_2(rebuild_answer), strict=False
            )
            review_answer = await review_assistant.ask(
                rebuild_answer_dict["repseudocode"]
            )
            print("\n" + "=" * 40)
            print("🤖 安全架构师 的回复")
            print("=" * 40)
            print(review_answer)


if __name__ == "__main__":
    asyncio.run(main())
