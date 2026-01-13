# 导入系统变量，主要为了获取不同环境API Key
from dotenv import load_dotenv

load_dotenv()

# 自定义本地嵌入函数，使用 SentenceTransformer
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):

        # 既然是处理文档，我们用这一款支持多语言的模型，效果比较均衡
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        # 列表推导式，把每个 numpy array 转成 list
        return [e.tolist() for e in embeddings]

    def name(self):
        return "local_sentence_transformer_v2"


# 连接数据库，定义好数据库地址，这里为rst/chroma_db
import chromadb
from google import genai
import os
import streamlit as st


@st.cache_resource
def get_chromadb_collection():
    return chromadb.PersistentClient(path="rst/chroma_db").get_collection(
        name="categorized_memory", embedding_function=LocalEmbeddingFunction()
    )


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


collection = get_chromadb_collection()
# Gemini API 客户端
gemini_client = get_gemini_client()

# --- 2. RAG 系统主函数，结合检索和生成 ---
from google.genai import types


def query_rag_system(user_query: str, category_filter: str) -> tuple[str, list]:
    # 1. 检索 (Retrieval)
    # 根据用户选择的模式，动态构造 where 过滤条件
    where_condition = {"category": category_filter}

    results = collection.query(
        # 假设只有一个问题
        query_texts=[user_query],
        n_results=3,  # 找 3 条证据
        where=where_condition,  # 🔥 Day 10 的核心魔法
    )

    # 2. 组装 Context
    valid_docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not valid_docs:
        return "没有找到相关资料。", []

    context_str = "\n".join(valid_docs)

    # 3. 生成 (Generation)
    system_prompt = """
    ###角色
    你是一个基于【私有知识库】的智能助手。
    ###任务
    请根据提供的上下文回答用户问题。
    ###约束
    1.如果上下文中没有答案，请诚实地说不知道，不要编造。
    2.区分“事件的次数”和“动作的频率”。例如，“打了三次球”通常意味着“这是一次打球的活动”。
    3.如果资料中没有明确提到见面的次数，请回答“资料中只提到了具体的互动细节，未统计见面总次数”。
    """

    user_prompt = f"""
    【参考资料 ({category_filter}类)】
    {context_str}
    
    【用户问题】
    {user_query}
    """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
            ),
            contents=[user_prompt],
        )
        return response.text, metadatas
    except Exception as e:
        return f"❌ 调用 Gemini 出错: {str(e)}", []


# --- 3. Streamlit UI 界面构建 ---


st.set_page_config(page_title="胖墩墩的知识库", page_icon="🐶", layout="wide")

# === 侧边栏：控制台 ===
with st.sidebar:
    st.image(
        "img/DSC01879.jpeg", caption="我是胖墩墩", use_container_width=True
    )  # 假设你上传了这张图
    st.title("🎛️ 模式选择")

    # 核心交互控件：选择查询范围
    mode = st.radio("你想查询什么？", ("🐶 胖墩墩的生活日记", "💻 赵一清的技术文档"))

    # 映射到数据库的 category 标签
    if "日记" in mode:
        category_filter = "diary"
        st.info("当前模式：只检索【日记】JSON 数据")
    else:
        category_filter = "tech"
        st.success("当前模式：只检索【技术】Markdown 文档")

    st.markdown("---")
    st.caption("Powered by zhao yi qing")

# === 主界面：聊天窗口 ===
st.title("💬 私有知识库助手")

# 初始化聊天历史 (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 接收用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. 调用 AI (显示加载动画)
    with st.chat_message("assistant"):
        with st.spinner(f"🔍 正在检索 {category_filter} 库..."):
            answer, sources = query_rag_system(prompt, category_filter)

            # 显示回答
            st.markdown(answer)

            # 显示引用来源 (折叠状态)
            if sources:
                with st.expander("📚 查看参考来源 (Evidence)"):
                    for i, meta in enumerate(sources):
                        # 动态展示 Metadata
                        source_name = meta.get("source", "未知文件")
                        author = meta.get("author", meta.get("subject", "未知"))
                        st.caption(
                            f"📄 来源 {i+1}: **{source_name}** (相关人: {author})"
                        )

    # 保存 AI 回复到历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
