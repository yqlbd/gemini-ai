# 将环境变量加载到系统环境中
import dotenv

dotenv.load_dotenv()

from google import genai
import os
import re
import json
import glob
import asyncio


# 代码生成助手
class CodeAssistant:
    # name为角色名称，system_instruction角色指令
    def __init__(self, name, system_instruction):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-2.0-flash"
        self.name = name
        self.system_instruction = system_instruction

    # JSON 清洗工具，用于返回干净的 JSON 字符串
    def clean_json_string(self, raw_text: str) -> str:
        """
        工业级清洗逻辑：
        1. 提取 JSON 块
        2. 允许控制字符（strict=False）
        """
        if not raw_text:
            return ""
        # 第一步：精准提取 Markdown 块
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.S | re.I)
        if match:
            clean_text = match.group(1)
        else:
            # 备选：提取第一个 { 和最后一个 } 之间的内容
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            clean_text = (
                raw_text[start : end + 1] if start != -1 and end != -1 else raw_text
            )

        # 第二步：只去除首尾空白，不干扰内部结构
        return clean_text.strip()

    # 给对应的角色发送消息，message为用户指令
    async def ask(self, message):
        print(f"⏳ 正在由{self.name}处理... (这是相当耗时的步骤，请耐心等待)")
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=[{"role": "user", "parts": [{"text": message}]}],
            config={"system_instruction": self.system_instruction},
        )
        # 进行清洗并返回
        return self.clean_json_string(response.text)


# 读取最新的JSON文件
@staticmethod
def load_latest_json(prefix: str = "project_tasks_db"):
    """
    自动寻找并读取最新的一份带有时间戳的 JSON 文件
    """
    # 获取所有匹配的文件列表
    files = glob.glob(f"rst/{prefix}_*.json")
    if not files:
        return None

    # 按文件名排序（因为带时间戳，最后一份就是最新的）
    latest_file = max(files, key=os.path.getctime)

    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f), latest_file


# 获取最大的score的对象
@staticmethod
def get_highest_score_item(review_list):
    highest_score = -1
    best_item = None
    for item in review_list:
        if item["score"] > highest_score:
            highest_score = item["score"]
            best_item = item
    return best_item


async def main():
    # 1.加载json文件
    history_data, file_name = load_latest_json("project_tasks_db")
    if history_data:
        print(f"✅ 已加载历史记忆: {file_name}")

        # 2.构造“带记忆”的提问
        context_prompt = f"""
        我们之前的设计方案如下：
        {json.dumps(history_data, ensure_ascii=False)}
        """
        context_prompt += """
        请基于上述架构，为其中的“分布式缓存扣减核心”模块编写一段伪代码实现。
        伪代码要求使用 Redis Lua 脚本，确保在高并发场景下不会出现超卖问题。
        严禁使用分布式锁等外部依赖，必须利用 Redis 自身的原子性操作来保证数据一致性。
        输出格式必须为 JSON，包含一个字段 "pseudocode"。
        示例：
        {
            "pseudocode": "local key = KEYS[1] local stock = redis.call('HGET', key, 'stock') if stock == nil or tonumber(stock) <= 0 then return 0 end local newStock = tonumber(stock) - 1 if newStock < 0 then return 0 end redis.call('HSET', key, 'stock', newStock) return 1"
        }
        """

        # 3.交给系统架构师
        sys_assistant = CodeAssistant(
            "系统架构师", "你是一个专业的系统架构师，擅长设计高可用、高性能的系统。"
        )
        sys_answer = await sys_assistant.ask(context_prompt)
        print("\n" + "=" * 40)
        print(f"🤖 {sys_assistant.name} 的回复")
        print("=" * 40)
        sys_answer_dict = json.loads(sys_answer, strict=False)
        print(sys_answer)
        # 4.交给安全架构师审计
        review_assistant = CodeAssistant(
            "安全架构师",
            """
            ### 角色：
            你是一个专门寻找并发漏洞的首席安全架构师，请对提交的Redis Lua脚本伪代码进行‘地狱级’审计。

            ### 背景：
            该 Lua 脚本用于在高并发环境下进行库存扣减，必须确保绝对不会出现超卖问题。

            ### 任务：
            1. 审查提交的伪代码，寻找潜在的并发安全漏洞。
            2. 重点寻找：竞态条件（Race Condition）、超卖风险、死锁、性能瓶颈、幂等性缺失。

            ### 约束：
            1. 如果发现没有使用 {} Hash Tag 或者没有处理原子性溢出，请直接给出 0 分并要求重构！

            ### 输出：
            1. 必须以 JSON 格式输出，包含 "score" (0-10分) 和 "critiques" (建议列表)

            ### 示例：
            {
                "score": 5,
                "critiques": ["Lua脚本未处理库存为负数的情况", "缺少分布式锁的过期保护"]
            }
            """,
        )
        review_answer = await review_assistant.ask(sys_answer_dict["pseudocode"])
        print("\n" + "=" * 40)
        print(f"🤖 {review_assistant.name} 的回复")
        print("=" * 40)
        review_answer_dict = json.loads(review_answer, strict=False)
        print(review_answer)

        review_list = []
        item = {
            "score": review_answer_dict["score"],
            "critiques": review_answer_dict["critiques"],
            "pseudocode": sys_answer_dict["pseudocode"],
        }
        review_list.append(item)
        # 5.根据审计结果决定是否重构
        gen_times = 1

        last_score = review_answer_dict["score"]
        while last_score < 7 and gen_times < 3:
            print("\n❌ 安全审计未通过，分数低于 7 分，要求重构！\n")
            # 交给重构AI
            rebuild_assistant = CodeAssistant(
                "重构工程师",
                """
                ### 角色：
                你是一个经验丰富的重构工程师，专门负责根据安全审计反馈对代码进行改进和优化。

                你的任务：
                1. 根据伪代码和安全审计架构师的反馈，进行代码重构。
                2. 输出JSON格式的数据，严禁包含任何自然语言的解释或开场白。
                3. 确保重构后的代码解决了所有审计中提到的问题，特别是并发安全和数据一致性方面的漏洞。

                JSON 结构输出示例：
                {
                    "pseudocode": "重构后的伪代码实现"
                }
                """,
            )
            rebuild_context = f"""
            我们之前的设计伪代码如下：
            {json.dumps(review_list, ensure_ascii=False)}
            请根据上述的内容中的代码和审核意见，进行重构，不要再次犯之前的错误！
            """
            print(f"⏳ 第 {gen_times} 次重构开始...")
            rebuild_answer = await rebuild_assistant.ask(rebuild_context)

            print("\n" + "=" * 40)
            print(f"🤖 {rebuild_assistant.name} 的回复")
            print("=" * 40)
            print(rebuild_answer)

            rebuild_answer_dict = json.loads(rebuild_answer, strict=False)
            # 重新交给安全架构师审计
            review_answer = await review_assistant.ask(
                rebuild_answer_dict["pseudocode"]
            )
            review_answer_dict = json.loads(review_answer, strict=False)
            # 获取分数
            last_score = review_answer_dict["score"]
            gen_times += 1

            print("\n" + "=" * 40)
            print(f"🤖 {review_assistant.name} 的回复")
            print("=" * 40)
            print(review_answer)

            # 记录本次重构结果
            review_list.append(
                {
                    "score": last_score,
                    "critiques": review_answer_dict["critiques"],
                    "pseudocode": rebuild_answer_dict["pseudocode"],
                }
            )

        final_item = get_highest_score_item(review_list)
        print("\n" + "=" * 40)
        print("✅ 最终结果")
        print("=" * 40)
        print(final_item)


if __name__ == "__main__":
    asyncio.run(main())
