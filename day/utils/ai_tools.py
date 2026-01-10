# utils/ai_tools.py
import re
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import glob
import os


class AIToolkit:
    """AI 开发助手工具箱"""

    @staticmethod
    def clean_json_string(raw_text: str) -> str:
        """
        清洗 AI 返回的 Markdown 格式字符串，提取 JSON 部分。
        """
        # 兼容有无 ```json 标签的情况
        match = re.search(r"```json\s+(.*?)\s+```", raw_text, re.S)
        clean_text = match.group(1) if match else raw_text.strip().replace("```", "")
        return clean_text.strip()

    @staticmethod
    def clean_json_string_2(raw_text: str) -> str:
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

    @staticmethod
    def save_to_json(data: dict, prefix: str = "project_tasks_db"):
        """
        保存 JSON，文件名带上精确到秒的时间戳
        格式示例: project_tasks_db_20260106150001.json
        """
        try:
            # 1. 生成时间戳字符串 (年月日时分秒)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # 2. 拼接完整文件名
            filename = f"{prefix}_{timestamp}.json"

            # 3. 执行写入
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            from rich.console import Console

            Console().print(f"\n[bold green]💾 数据已持久化至: {filename}[/bold green]")

            return filename  # 返回文件名，方便主程序记录

        except Exception as e:
            print(f"保存文件时发生错误: {e}")

    @staticmethod
    def load_latest_json(prefix: str = "project_tasks_db"):
        """
        自动寻找并读取最新的一份带有时间戳的 JSON 文件
        """
        # 获取所有匹配的文件列表
        files = glob.glob(f"{prefix}_*.json")
        if not files:
            return None

        # 按文件名排序（因为带时间戳，最后一份就是最新的）
        latest_file = max(files, key=os.path.getctime)

        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f), latest_file

    @staticmethod
    def print_tasks_table(data: dict):
        """
        使用 Rich 库打印漂亮的任务表格
        """
        console = Console()

        # 打印标题
        console.print(
            Panel(
                f"[bold magenta]{data.get('project_name', '未命名项目')}[/bold magenta]",
                title="项目任务清单",
            )
        )

        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("模块", style="yellow", width=20, overflow="fold")
        table.add_column("优先级", justify="center", width=10)
        table.add_column("耗时", style="blue", width=8)
        table.add_column("任务详情", style="white")

        for item in data.get("details", []):
            tasks_str = "\n".join([f"• {t}" for t in item.get("tasks", [])])
            p_color = {"高": "red", "中": "yellow", "低": "green"}.get(
                item.get("priority"), "white"
            )

            table.add_row(
                item.get("module", "N/A"),
                f"[{p_color}]{item.get('priority', 'N/A')}[/{p_color}]",
                item.get("estimated_time", "N/A"),
                tasks_str,
            )
            table.add_section()

        console.print(table)
