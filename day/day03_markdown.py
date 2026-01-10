import asyncio
import os
from google import genai
from datetime import datetime

# 1. 引入 dotenv 并加载环境变量
from dotenv import load_dotenv
load_dotenv()

class ProjectAssistant:
    def __init__(self):
        # 🔑 核心升级：植入 CoT 思维链指令
        self.system_instruction = """
        你是一个有丰富经验的系统架构师，擅长一步步思考。
        
        在回答之前，请务必遵循以下步骤 (Chain of Thought)：
        1. 【场景分析】：深入分析用户的具体业务场景、并发量级和核心约束条件。
        2. 【瓶颈识别】：指出在当前场景下，系统最可能挂在哪个环节（如数据库死锁、带宽打满）。
        3. 【方案设计】：给出分层防御的架构方案（如网关层、缓存层、数据库层），并解释技术选型理由。
        4. 【兜底策略】：设计异常情况下的降级或熔断方案。
        
        输出格式要求：请使用清晰的 Markdown 格式，包含标题、加粗和代码块。
        """
        
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-3-pro-preview" # 使用更强大的 Gemini 3 Pro 模型
        # 手动管理历史记录
        self.history = [] 

    async def ask(self, message):
        """发送消息并获取回复 (Async)"""
        print(f"⏳ 正在思考架构方案... (这是最耗时的步骤，请耐心等待)")
        
        # 构造请求：历史 + 当前问题
        messages = self.history + [{"role": "user", "parts": [{"text": message}]}]
        
        # 调用 API
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=messages,
            config={'system_instruction': self.system_instruction}
        )
        
        # 更新历史
        self.history.append({"role": "user", "parts": [{"text": message}]})
        self.history.append({"role": "model", "parts": [{"text": response.text}]})
        
        return response.text

    def save_to_markdown(self, content, filename="rst/architecture_design.md"):
        """将结果保存为文件"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n✅ 文档已保存至本地: {filename}")

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
    print("\n" + "="*40)
    print("🤖 架构师 Gemini 的回复 (预览前500字):")
    print("="*40)
    print(answer[:500] + "...\n(内容太长，已省略后续)")
    
    # 3. 保存完整文档
    architect.save_to_markdown(answer)

if __name__ == "__main__":
    asyncio.run(main())