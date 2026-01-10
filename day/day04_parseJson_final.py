from google import genai
import dotenv
import os
import json
import asyncio
from utils.ai_tools import AIToolkit  # 从工具包导入类

dotenv.load_dotenv


class ProjectAssistant:
    def __init__(self):
        # 🔑 核心升级：植入 CoT 思维链指令
        self.system_instruction = """
        ### 角色：
        你是一个精通项目管理的系统架构师，对项目的健壮性和可靠性有着近乎偏执的追求，愿意花时间打磨产品，不为上线时间妥协。
        """
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-3-pro-preview"  # 使用更强大的 Gemini 3 Pro 模型
        # 手动管理历史记录
        self.history = []

    async def ask(self, message):
        """发送消息并获取回复 (Async)"""
        print(f"⏳ 正在由系统架构师处理... (这是非常耗时的步骤，请耐心等待)")

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


async def main():
    # 实例化助手
    architect = ProjectAssistant()

    # 🎯 你的真实业务需求
    user_requirement = """
    ### 背景：
    我想做一个理财产品的秒杀方案。请将以下秒杀系统的需求拆解为开发任务。
    - 时间：每天9点准时开抢
    - 额度：100w
    - 预估并发：瞬间用户 50,000 人
    - 持续时间：预计 5s 内抢完
    ### 约束条件
    严禁超卖，这是金融产品。
    ### 格式要求：
    1. 必须只返回 JSON 格式数据，严禁包含任何自然语言的解释或开场白。
    2. 必须严格遵守以下 JSON Schema（结构定义）：
    {
        "type": "object",
        "properties": {
            "project_name": { "type": "string", "description": "项目名称" },
            "total_modules": { "type": "integer", "description": "模块的总计数量" },
            "details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "module": { "type": "string" },
                "priority": { "type": "string", "enum": ["高", "中", "低"] },
                "estimated_time": { "type": "string", "pattern": "^\\d+天$" },
                "tasks": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["module", "priority", "estimated_time", "tasks"]
            }
        },
        "required": ["project_name", "total_modules", "details"]
    }
    ### 范例 (Few-Shot)：
    用户需求：设计一个基础的 Redis 库存扣减模块。
    输出：
    {
        "project_name": "Redis 库存模块",
        "total_modules": 1,
        "details": [{
            "module": "缓存层",
            "priority": "高",
            "estimated_time": "1天",
            "tasks": [
                "编写 Lua 脚本实现 DECR 原子扣减与库存校验",
                "配置 Redis 连接池，设置合理的超时重试策略",
                "实现库存预热脚本，将 MySQL 数据同步至 Redis"
            ]
        }]
    }
    """

    print(f"🚀 发送需求: {user_requirement.strip().splitlines()[0]}...")

    # 1. 异步调用 AI
    answer = await architect.ask(user_requirement)

    # 2. 打印部分结果到屏幕
    print("\n" + "=" * 40)
    print("🤖 架构师 Gemini 的回复")
    print("=" * 40)
    # print(answer)
    # 3. 清理并解析 JSON
    final_dict = json.loads(AIToolkit.clean_json_string(answer))
    print("\n✅ 已成功解析为 JSON 格式！")
    # 调用工具箱中的方法打印表格和保存 JSON
    AIToolkit.print_tasks_table(final_dict)
    AIToolkit.save_to_json(final_dict, "project_tasks_db")


if __name__ == "__main__":
    asyncio.run(main())
