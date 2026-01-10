import asyncio
from google import genai
from PIL import Image
import os
from pathlib import Path


async def analyze_dondon():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # 1. 加载胖墩墩的照片 (记得替换为你电脑里的实际路径)
    # --- 修改部分：获取相对路径 ---
    # 获取当前脚本所在目录的绝对路径
    current_dir = Path(__file__).parent 
    # 2. 获取上级项目根目录 (my_project)，然后进入 img 文件夹
    image_path = current_dir.parent / "img" / "DSC01879.jpeg"
    print(f"📂 尝试读取路径: {image_path}")
    img = Image.open(image_path)

    prompt = """
    你是一个宠物行为心理学家。请观察这张照片中的狗狗（它叫胖墩墩）：
    1. 识别它的品种和目前的肢体语言（眼神、耳朵、尾巴姿势）。
    2. 判断它现在的心情（开心、委屈、想睡觉、还是想出去玩？）。
    3. 给出详细的照顾建议。
    最后请以 JSON 格式输出结果。
    """

    print("⏳ 正在让 Gemini 观察胖墩墩...")
    # 注意：contents 是一个列表，可以同时包含文字和图片对象
    response = await client.aio.models.generate_content(
        model="gemini-2.0-flash", contents=[prompt, img]  # 这里直接放入 Image 对象
    )

    print("\n🐶 胖墩墩的情绪报告：")
    print(response.text)


if __name__ == "__main__":
    asyncio.run(analyze_dondon())
