from google import genai
from google.genai import types
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. 初始化配置 ---

# 初始化新版 Google Client
# 注意：这里我们用同步客户端 (client)，而不是异步 (client.aio)，方便脚本直接运行
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# 初始化本地 Embedding 模型 (复用昨天的逻辑)
class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        print("🤖 正在加载本地 Embedding 模型...")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        return [e.tolist() for e in embeddings]

    def name(self):
        return "local_sentence_transformer_v2"


# 连接 ChromaDB
print("💾 正在连接记忆库...")
chroma_client = chromadb.PersistentClient(path="rst/chroma_db")
collection = chroma_client.get_collection(
    name="pangdundun_memory", embedding_function=LocalEmbeddingFunction()
)

# --- 2. 定义系统指令 (System Instruction) ---
# 💡 新版 SDK 的强项：把“人设”和“约束”放在系统层级，权重更高！
SYSTEM_PROMPT = """
你是一个专门帮助用户的 AI 助手。
你的任务是根据用户提供的【上下文记忆】来回答问题。

### 核心约束
1. 严禁编造：你必须且只能依据【上下文记忆】中的信息回答。
2. 诚实原则：如果【上下文记忆】里没有包含问题的答案，请直接回答：“根据现有的资料，我没找到这个问题的答案。”，不要尝试用通识知识去瞎猜。
3. 语气风格：请用温柔、简洁的中文回答。
"""


def rag_chat(user_query):
    print(f"\n🔍 正在检索关于 '{user_query}' 的记忆...")

    # --- 3. 检索 ChromaDB ---
    # 我们找 3 条最相关的
    results = collection.query(query_texts=[user_query], n_results=3)

    # 数据清洗：处理检索结果
    documents = results["documents"][0]
    distances = results["distances"][0]

    # 简单的阈值过滤 (比如距离 > 30 就不看了，防止噪声)
    valid_docs = []
    for doc, dist in zip(documents, distances):
        if dist < 30:  # 这里的30是经验值，根据不同模型调整
            valid_docs.append(doc)

    if not valid_docs:
        print("❌ 没找到相关记忆 (可能是距离太远被过滤了)")
        context_str = "无"
    else:
        print(f"✅ 找到了 {len(valid_docs)} 条有效记忆！")
        # 把列表拼成一个长字符串
        context_str = "\n".join(valid_docs)
        print(f"📖 相关记忆内容如下：\n{context_str}")

    # --- 4. 组装用户 Prompt ---
    # 这里只放“动态内容”：上下文 + 问题
    user_prompt = f"""
    【上下文记忆】
    {context_str}

    【用户问题】
    {user_query}
    """

    print("🧠 AI 正在结合内容思考...")

    # --- 5. 调用新版 SDK ---
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # 建议用 1.5-flash 或 2.0-flash-exp，速度快
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,  # 👈 刚才定义的约束放这里
                temperature=0.3,  # 让回答更严谨，越低越不容易胡编
            ),
            contents=[user_prompt],
        )
        return response.text
    except Exception as e:
        return f"❌ 调用出错: {str(e)}"


# --- 主程序入口 ---
if __name__ == "__main__":

    # 测试问题 2：找不到答案的 (测试防幻觉)
    q2 = "胖墩墩会写代码吗？"
    print(f"🤖 最终回答: {rag_chat(q2)}")
    print("-" * 30)

    # 测试问题 1：能找到答案的
    q1 = "胖墩墩喜欢吃什么？"
    print(f"🤖 最终回答: {rag_chat(q1)}")

    print("-" * 30)

    # 测试问题 3：
    q3 = "胖墩墩喜欢洗澡吗？"
    print(f"🤖 最终回答: {rag_chat(q3)}")

    q4 = "赵一清会写代码吗？"
    print(f"🤖 最终回答: {rag_chat(q4)}")
