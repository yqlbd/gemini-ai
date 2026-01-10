import os
import re
import json
import glob
import asyncio
import dotenv
from datetime import datetime
from google import genai

# 加载环境变量 (确保 GEMINI_API_KEY 已配置)
dotenv.load_dotenv()


# ==========================================
# 1. 通用 Agent 助手类
# ==========================================
class CodeAssistant:
    def __init__(self, name, system_instruction):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        # 使用 gemini-2.0-flash，速度快且逻辑严谨，非常适合多轮审计
        self.model_id = "gemini-2.0-flash"
        self.name = name
        self.system_instruction = system_instruction

    def clean_json_string(self, raw_text: str) -> str:
        """强化版 JSON 清洗：提取、转义处理与容错"""
        if not raw_text:
            return ""

        # 提取 ```json ... ``` 中的内容
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.S | re.I)
        clean_text = match.group(1) if match else raw_text

        if not match:
            # 备选：截取第一个 { 和最后一个 }
            start = clean_text.find("{")
            end = clean_text.rfind("}")
            clean_text = (
                clean_text[start : end + 1] if start != -1 and end != -1 else clean_text
            )

        return clean_text.strip()

    async def ask(self, message):
        print(f"⏳ [{self.name}] 正在思考中...")
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=[{"role": "user", "parts": [{"text": message}]}],
                config={"system_instruction": self.system_instruction},
            )
            return self.clean_json_string(response.text)
        except Exception as e:
            print(f"❌ [{self.name}] API 调用失败: {e}")
            return None


# ==========================================
# 2. 静态工具函数
# ==========================================
def load_latest_json(prefix: str = "project_tasks_db"):
    files = glob.glob(f"{prefix}_*.json")
    if not files:
        return None, None
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f), latest_file


# ==========================================
# 3. 角色 Instruction 定义
# ==========================================
ARCHITECT_PROMPT = """你是一个专业的系统架构师。请为 Redis 库存扣减服务设计 Lua 脚本。
必须包含 'pseudocode' 字段。要求：高并发原子性、幂等性、支持 Redis Cluster (Hash Tag)。"""

AUDITOR_PROMPT = """你是一个毒舌的安全架构师。请审计 Lua 脚本。
重点：1. 是否有 {} Hash Tag 解决 CROSSSLOT？ 2. 是否校验了扣减数量为正数？ 3. 变量空值处理。
输出 JSON 格式：{"score": 0-10, "critiques": ["建议1", "建议2"]}。
注意：不要在回复中包含任何非 JSON 内容，字符串内严禁使用未转义的双引号。"""

REFACTOR_PROMPT = """你是一个重构工程师。根据历史评价优化代码。
你的目标是修正所有安全漏洞，并确保代码在 Redis Cluster 环境下 100% 可用。
输出 JSON 格式：{"pseudocode": "重构后的代码"}"""


# ==========================================
# 4. 主程序逻辑
# ==========================================
async def main():
    # 1. 环境准备
    history_data, file_name = load_latest_json("project_tasks_db")
    if not history_data:
        print("❌ 未找到项目设计 JSON 文件，请检查路径。")
        return
    print(f"✅ 已加载架构背景: {file_name}")

    # 2. 实例化 Agent 团队
    architect = CodeAssistant("系统架构师", ARCHITECT_PROMPT)
    auditor = CodeAssistant("安全审计师", AUDITOR_PROMPT)
    refactor = CodeAssistant("重构工程师", REFACTOR_PROMPT)

    # 3. 初始化迭代参数
    history_records = []  # 存储每一轮的结果：{"score": 0, "code": "", "critiques": []}
    max_iterations = 5
    current_code = ""

    # 第一步：由架构师生成初始代码
    print("\n--- 🏗️ 第一步：架构师生成初始方案 ---")
    arch_raw = await architect.ask(
        f"参考设计：{json.dumps(history_data, ensure_ascii=False)}"
    )
    try:
        current_code = json.loads(arch_raw, strict=False).get("pseudocode", "")
    except Exception:
        print("❌ 初始代码解析失败，请检查 API 输出。")
        return

    # 4. 进入演化循环
    for i in range(max_iterations):
        print(f"\n--- 🔄 第 {i+1} 轮进化开始 ---")

        # A. 审计阶段
        audit_raw = await auditor.ask(f"请审计此代码：\n{current_code}")

        try:
            # 增加容错：如果 JSON 解析失败，尝试强制修复或给低分触发重构
            audit_res = json.loads(audit_raw, strict=False)
            score = audit_res.get("score", 0)
            critiques = audit_res.get("critiques", [])
        except json.JSONDecodeError as e:
            print(f"⚠️ 审计结果 JSON 解析异常，本轮判为 0 分。错误: {e}")
            score = 0
            critiques = [
                "JSON 解析失败，说明输出格式混乱，请重新重构并确保输出纯净 JSON。"
            ]

        # B. 记录当前版本
        history_records.append(
            {
                "round": i + 1,
                "score": score,
                "code": current_code,
                "critiques": critiques,
            }
        )
        print(f"📊 本轮审计得分: {score}")

        # C. 退出条件：满分或达到轮数
        if score >= 9:
            print("✨ 代码已近完美，提前结束迭代。")
            break

        # D. 重构阶段 (如果不满 9 分且还有轮数)
        if i < max_iterations - 1:
            # 将所有历史评价喂给重构工程师，让它“长记性”
            refactor_msg = f"历史代码：\n{current_code}\n\n历史审计意见：\n{json.dumps(critiques, ensure_ascii=False)}"
            refactor_raw = await refactor.ask(refactor_msg)
            try:
                current_code = json.loads(refactor_raw, strict=False).get(
                    "pseudocode", ""
                )
            except:
                print("⚠️ 重构代码解析异常，沿用上一版进行下一轮。")

    # 5. 优中选优：从 history_records 中选出分数最高的一版
    best_version = max(history_records, key=lambda x: x["score"])

    print("\n" + "🏆" * 20)
    print(f"最终获胜版本：第 {best_version['round']} 轮迭代成果")
    print(f"最高得分：{best_version['score']}")
    print("🏆" * 20)
    print(best_version["code"])

    # 6. 持久化存储最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"optimized_redis_service_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(best_version, f, ensure_ascii=False, indent=4)
    print(f"\n🔥 最终方案已保存至: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
