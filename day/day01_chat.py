import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class ProjectAssistant:
    def __init__(self, system_instruction):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-2.0-flash-exp"
        self.system_instruction = system_instruction
        # 1. 我们手动管理历史记录，不依赖 SDK 内部属性
        self.history = [] 

    def ask(self, message):
        # 2. 构造对话
        # 我们直接使用 client.models.generate_content，手动传递 history
        # 这是最符合“无状态”转换“有状态”的逻辑
        
        # 组装当前的请求内容：历史记录 + 当前问题
        messages = self.history + [{"role": "user", "parts": [{"text": message}]}]
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=messages,
            config={'system_instruction': self.system_instruction}
        )
        
        # 3. 更新历史记录 (保存这一轮的一问一答)
        self.history.append({"role": "user", "parts": [{"text": message}]})
        self.history.append({"role": "model", "parts": [{"text": response.text}]})
        
        return response.text

def main():
    instruction = "你是一个架构师，说话简洁。你现在的任务是协助一清开发一个 AI 项目。"
    assistant = ProjectAssistant(instruction)

    print("🚀 稳健版助手已就绪（输入 'quit' 退出）")
    
    while True:
        user_input = input("一清 > ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        try:
            answer = assistant.ask(user_input)
            print(f"\n🤖 Gemini > {answer}\n")
            print(f"📊 当前对话轮数: {len(assistant.history) // 2}")
        except Exception as e:
            print(f"❌ 出错啦: {e}")

if __name__ == "__main__":
    main()