import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import re
import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import jieba
from hot_questions_manager import HotQuestionsManager

# ===== 配置 DeepSeek API =====
DEEPSEEK_API_KEY = ""  # 替换成你的Key
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# ===== 加载 CSV 数据（红旗 + 竞品） =====
def load_car_data():
    try:
        df_hongqi = pd.read_csv("data/hongqi_data.csv", encoding="utf-8")
        df_hongqi["品牌"] = "红旗"
    except Exception as e:
        st.error(f"读取红旗数据失败：{e}")
        return None

    try:
        df_competitor = pd.read_csv("data/competitor_data.csv", encoding="utf-8")
    except Exception as e:
        st.warning(f"竞品数据未找到，只使用红旗数据：{e}")
        df_competitor = pd.DataFrame()

    df = pd.concat([df_hongqi, df_competitor], ignore_index=True)
    df = df.fillna("")
    print(f"已加载 {len(df_hongqi)} 条红旗数据 + {len(df_competitor)} 条竞品数据 = {len(df)} 条总计")
    return df


# ===== 构建文本列表 =====
def build_texts(df):
    texts = []
    for _, row in df.iterrows():
        text = f"品牌：{row['品牌']} 车型：{row['车型']} 指导价{row['指导价(万元)']}万元 级别：{row['级别']} 能源：{row['能源类型']} 发动机：{row['发动机']} 配置：{row['主要配置']}"
        texts.append(text)
    return texts


# ===== 向量化 =====
def build_vector_index(df):
    texts = build_texts(df)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors, texts


# ===== BM25 =====
def build_bm25_index(df, texts):
    tokenized_texts = [list(jieba.cut(text)) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    return bm25


# ===== 向量检索 =====
def vector_search(query, vectorizer, vectors, df, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices].copy()
    results['相似度'] = similarities[top_indices]
    return results


# ===== BM25 检索 =====
def bm25_search(query, bm25, df, texts, top_n=5):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = df.iloc[top_indices].copy()
    results['bm25分数'] = [scores[i] for i in top_indices]
    return results


# ===== 检索（数值排序兜底 + 多路召回） =====
def search_car(query, vectorizer, vectors, bm25, df, texts, top_n=3):
    numeric_keywords = {
        "最贵": ("指导价(万元)", False),
        "最便宜": ("指导价(万元)", True),
        "最高": ("指导价(万元)", False),
        "最低": ("指导价(万元)", True),
        "最大": ("轴距(mm)", False),
        "最长": ("轴距(mm)", False),
        "最轻": ("整备质量(kg)", True),
        "最重": ("整备质量(kg)", False),
    }

    for keyword, (field, ascending) in numeric_keywords.items():
        if keyword in query:
            if field in df.columns:
                full_df = df.copy()
                full_df[field] = pd.to_numeric(full_df[field], errors='coerce')
                full_df = full_df.dropna(subset=[field])
                full_df = full_df.sort_values(by=field, ascending=ascending)
                result_df = full_df.head(top_n)
                print(f"触发全局排序：{keyword} -> {field} {'升序' if ascending else '降序'}")
                return result_df, texts

    vec_results = vector_search(query, vectorizer, vectors, df, top_n=5)
    bm25_results = bm25_search(query, bm25, df, texts, top_n=5)

    merged_indices = []
    for idx in bm25_results.index:
        if idx not in merged_indices:
            merged_indices.append(idx)
    for idx in vec_results.index:
        if idx not in merged_indices:
            merged_indices.append(idx)

    result_df = df.iloc[merged_indices].head(top_n)
    return result_df, texts


# ===== 生成回答 =====
def generate_response(query, results_df, texts):
    if results_df is None or results_df.empty:
        return "抱歉，没有找到相关的车型信息。你可以试试问具体的车型，比如：红旗H9多少钱？"

    context = "以下是相关车型的参数信息：\n\n"
    for _, row in results_df.iterrows():
        context += f"品牌：{row['品牌']} 车型：{row['车型']} {row['版本']}\n"
        context += f"指导价：{row['指导价(万元)']}万元\n"
        context += f"级别：{row['级别']}\n"
        context += f"能源类型：{row['能源类型']}\n"
        context += f"发动机：{row['发动机']}\n"
        context += f"尺寸：{row['长*宽*高(mm)']}\n"
        context += f"轴距：{row['轴距(mm)']}\n"
        context += f"主要配置：{row['主要配置']}\n"
        context += "---\n\n"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": """你是红旗汽车的AI销售顾问，必须严格基于提供的车型数据回答。
                数据中既有红旗车型，也有竞品车型。
                当用户问红旗车型时，重点介绍红旗的优势。
                当用户问对比时，客观列出参数差异，突出红旗的性价比和配置优势。
                如果参考数据中没有相关信息，请明确告诉用户没有该数据，不要编造。
                回答要专业、热情、简洁。价格直接说指导价，并建议到店咨询优惠。"""},
                {"role": "user", "content": f"用户问题：{query}\n\n参考数据：\n{context}"}
            ],
            stream=True
        )
        return response
    except Exception as e:
        return f"调用API失败：{str(e)}"


# ===== Streamlit UI =====
st.set_page_config(page_title="红旗汽车智能销售助手", page_icon="🚗", layout="wide")

st.markdown("""
<style>
    .main-header { color: #8B0000; font-size: 2.2rem; font-weight: 700; text-align: center; padding: 0.5rem 0 0.2rem 0; }
    .sub-header { text-align: center; color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
    .stButton button { background-color: #8B0000; color: white; font-weight: 500; border-radius: 20px; padding: 0.3rem 1.2rem; border: none; }
    .stButton button:hover { background-color: #a00000; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚗 红旗汽车智能销售助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于向量检索技术，准确理解你的需求，从车型知识库中找到最相关的信息</div>', unsafe_allow_html=True)

# ===== 加载数据 =====
df = load_car_data()
if df is None:
    st.stop()

texts = build_texts(df)
vectorizer, vectors, _ = build_vector_index(df)
bm25 = build_bm25_index(df, texts)

# ===== 初始化热门问题管理器 =====
hot_mgr = HotQuestionsManager()

# ===== 侧边栏 =====
with st.sidebar:
    st.markdown("### 📊 系统状态")
    hongqi_count = len(df[df['品牌'] == '红旗']) if '品牌' in df.columns else len(df)
    st.success(f"✅ 已加载 {len(df)} 款车型（红旗 {hongqi_count} 款 + 竞品 {len(df) - hongqi_count} 款）")
    st.info("🧠 向量引擎就绪（TF-IDF + BM25）")

    st.markdown("---")
    st.markdown("### 🔥 热门问题")
    st.markdown("点击即可快速提问：")

    # 获取热门问题
    top_questions = hot_mgr.get_top_questions(5)

    # 去重：用 set 记录已有问题
    seen_questions = set()
    deduplicated = []

    # 先加真实热门问题
    for q_label in top_questions:
        q_text = re.sub(r"（热度：\d+）", "", q_label).strip()
        if q_text not in seen_questions:
            seen_questions.add(q_text)
            deduplicated.append(q_label)

    # 如果少于3个，加默认问题（也去重）
    if len(deduplicated) < 3:
        default_questions = [
            "红旗H9多少钱",
            "哪款车适合二胎家庭",
            "红旗H9和奥迪A6L哪个好"
        ]
        for q in default_questions:
            if q not in seen_questions:
                seen_questions.add(q)
                deduplicated.append(f"{q}（热度：0）")
            if len(deduplicated) >= 5:
                break

    # 显示按钮（使用索引保证 key 唯一）
    for idx, q_label in enumerate(deduplicated[:5]):
        q_text = re.sub(r"（热度：\d+）", "", q_label).strip()
        if st.button(q_label, key=f"hot_{q_text}_{idx}", use_container_width=True):
            st.session_state.quick_question = q_text

# ===== 初始化 session_state =====
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是红旗汽车的AI销售顾问，请问有什么可以帮你？"}
    ]
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ===== 处理热门问题点击 =====
if st.session_state.get("quick_question"):
    st.session_state.pending_question = st.session_state.quick_question
    st.session_state.quick_question = None

# ===== 显示聊天记录 =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===== 用户输入 =====
if st.session_state.get("pending_question"):
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    user_input = st.chat_input("请输入你关心的问题...")

if user_input:
    # 记录这个问题到热门问题管理器
    hot_mgr.record_question(user_input)

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 检索
    results_df, _ = search_car(user_input, vectorizer, vectors, bm25, df, texts)

    # 生成回答
    response_stream = generate_response(user_input, results_df, texts)

    if hasattr(response_stream, '__iter__') and not isinstance(response_stream, str):
        with st.chat_message("assistant"):
            full_response = st.write_stream(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        with st.chat_message("assistant"):
            st.markdown(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": response_stream})

    st.rerun()