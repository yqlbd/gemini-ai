import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. 加载环境变量 (最安全的做法)
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. 定义系统指令：设定 AI 的身份和行为准则
SYSTEM_PROMPT = """
你是一个拥有 20 年经验的资深 Python 架构师。
你的任务是评审用户提交的代码。
要求：
1. 只指出代码中的逻辑缺陷、性能瓶颈或不规范命名。
2. 评价必须刻薄但精准。
3. 必须给出优化后的 Python 代码块。
4. 使用 Markdown 格式输出。
"""

def main():
    # 3. 在初始化模型时注入 System Instruction
    # 注意：在 V2 SDK 中，system_instruction 是放在 generate_content 的 config 里的
    
    user_code = """
    def total(n):
        res = 0
        for i in range(len(n)):
            res = res + n[i]
        return res
    """

    print("🧐 架构师正在审阅你的代码...\n")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=f"评审这段代码：\n{user_code}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3 # 调低随机性，让评审更严谨
        )
    )

    print(response.text)
    # 打印 Token 使用情况
    print(f"\n📈 统计: 输入 {response.usage_metadata.prompt_token_count} Tokens, "
      f"输出 {response.usage_metadata.candidates_token_count} Tokens")

if __name__ == "__main__":
    main()