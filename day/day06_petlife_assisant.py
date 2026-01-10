from day05_retrybuild import CodeAssistant

import json
from datetime import datetime
from PIL import Image
import asyncio
from pathlib import Path


# 继承你写的 CodeAssistant
class PetLifeAssistant(CodeAssistant):
    def __init__(self, name, system_instruction):
        super().__init__(name, system_instruction)
        self.log_prefix = "dog_life_log"

    async def analyze_and_log(self, image_path):
        """核心流：拍照 -> 识别 -> 存盘"""
        # 1. 加载图片
        try:
            img = Image.open(image_path)
        except Exception as e:
            return f"❌ 图片加载失败: {e}"

        # 2. 调用多模态接口
        print(f"⏳ {self.name} 正在深度观察照片中的胖墩墩...")
        prompt = "请根据这张照片，以 JSON 格式输出。"

        response_text = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=[prompt, img],
            config={"system_instruction": self.system_instruction},
        )

        # 3. 清洗并解析 JSON
        clean_json = self.clean_json_string(response_text.text)
        data = json.loads(clean_json, strict=False)

        # 4. 自动持久化存储 (增加时间戳)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"rst/{self.log_prefix}_{timestamp}.json"

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ 日志已成功同步至本地: {file_name}")
        return data


# --- 执行入口 ---
async def run_pet_care():
    # 角色定义
    instruction = """
        # 角色
        你是一个精通犬类行为学的宠物管家，请通过图片精准识别狗狗情绪并给出专业建议。

        # 背景
        胖墩墩是一只非常可爱的狗狗，主人非常关心它的情绪和健康状况，希望你能帮助他们更好地照顾胖墩墩。

        # 任务
        通过用户给出的照片，分析胖墩墩的品种、情绪状态。

        # 约束内容
        1.如果照片中没有狗狗，请说明并提示用户重新拍摄。

        # 输出格式
        请严格以下以下 JSON 格式输出结果，输出中文，不要加其他开场白或解释。

        # 示例
        识别出：
        {
            "dog_name": "胖墩墩",
            "breed_guess": "拉布拉多寻回犬",
            "helth_status": "良好",
            "happiness_level": 8,
            "mood_assessment": "开心且放松",
            "care_suggestions": "继续保持规律的运动和饮食，定期进行健康检查。"
        }

        未识别出：
        {
            "error": "照片中未检测到狗狗，请重新拍摄一张包含狗狗的照片。"
        }
        """

    steward = PetLifeAssistant("胖墩墩专属管家", instruction)

        # 1. 加载胖墩墩的照片 (记得替换为你电脑里的实际路径)
    # --- 修改部分：获取相对路径 ---
    # 获取当前脚本所在目录的绝对路径
    current_dir = Path(__file__).parent 
    # 2. 获取上级项目根目录 (my_project)，然后进入 img 文件夹
    image_path = current_dir.parent / "img" / "DSC01879.jpeg"
    # 假设你刚刚拍的照片
    await steward.analyze_and_log(image_path)
# 读取所有狗狗生活日志
def load_json_from_file(prefix: str = "dog_life_log"):
    """
    自动寻找并读取最新的一份带有时间戳的 JSON 文件
    """
    import glob
    import os

    dog_life_list = []
    # 获取所有匹配的文件列表
    files = glob.glob(f"{prefix}_*.json")
    if not files:
        return dog_life_list
    for dog_file in files:
        with open(dog_file, "r", encoding="utf-8") as f:
            dog_life_list.append(json.load(f))
    return dog_life_list


if __name__ == "__main__":
    asyncio.run(run_pet_care())

    # 读取所有狗狗生活日志
    # dog_life_list = load_json_from_file("dog_life_log")
    # # 组成df
    # import pandas as pd

    # df = pd.DataFrame(dog_life_list)
    # # 获取所有名字为“胖墩墩”的狗狗的平均快乐指数
    # print(
    #     f"胖墩墩平均快乐指数: {df[df['dog_name'] == '胖墩墩']['happiness_level'].mean()}"
    # )
