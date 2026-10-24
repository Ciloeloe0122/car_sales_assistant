import os
import csv
import json
import uuid
import re
from datetime import datetime, timedelta
from collections import defaultdict

from crm import get_crm, CRM

# 在应用启动时获取 CRM 实例
crm = get_crm()

import streamlit as st
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv
import markdown
from hot_questions_manager import HotQuestionsManager
import streamlit as st


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 配置 DeepSeek API ====================
try:
    DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

if not DEEPSEEK_API_KEY:
    st.error("未配置 DeepSeek API Key，请在 Secrets 中添加 DEEPSEEK_API_KEY")
    st.stop()

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)



# ==================== 数据加载与索引构建 ====================
@st.cache_data
def load_car_data():
    import csv

    # 构建绝对路径
    hongqi_path = os.path.join(BASE_DIR, "data", "hongqi_data.csv")
    competitor_path = os.path.join(BASE_DIR, "data", "competitor_data.csv")

    # 读取红旗数据（用 pandas 读，因为格式相对规范）
    try:
        df_hongqi = pd.read_csv("data/hongqi_data.csv", encoding="utf-8", dtype={"长*宽*高(mm)": str})
        df_hongqi["品牌"] = "红旗"
    except Exception as e:
        st.error(f"读取红旗数据失败：{e}")
        return None

    # 读取竞品数据（用原生 csv 模块，更健壮）
    try:
        rows = []
        with open("data/competitor_data.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头

            # 清理表头中的空格
            header = [col.strip() for col in header]

            for row in reader:
                # 如果这一行的列数少于表头，跳过
                if len(row) < len(header):
                    continue
                # 如果列数多于表头，尝试合并多余的列到最后一列
                if len(row) > len(header):
                    # 把多余的列合并到主要配置列（索引 13）
                    extra = row[13:]
                    row = row[:13] + ["，".join(extra)] + row[14:]
                # 确保列数一致
                if len(row) >= len(header):
                    rows.append(row[:len(header)])

        # 构建 DataFrame
        df_competitor = pd.DataFrame(rows, columns=header)
        if "品牌" not in df_competitor.columns:
            df_competitor["品牌"] = "竞品"

    except Exception as e:
        st.warning(f"竞品数据加载失败，只使用红旗数据：{e}")
        df_competitor = pd.DataFrame()

    # 合并数据
    if not df_competitor.empty:
        df = pd.concat([df_hongqi, df_competitor], ignore_index=True)
    else:
        df = df_hongqi
    df = df.fillna("")
    return df

def build_texts(df):
    texts = []
    for _, row in df.iterrows():
        text = f"品牌：{row['品牌']} 车型：{row['车型']} 指导价{row['指导价(万元)']}万元 级别：{row['级别']} 能源：{row['能源类型']} 发动机：{row['发动机']} 配置：{row['主要配置']}".lower()
        texts.append(text)
    return texts


@st.cache_resource
def build_vector_index(df):
    texts = build_texts(df)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors, texts


@st.cache_resource
def build_bm25_index(df, texts):
    tokenized_texts = [list(jieba.cut(text)) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    return bm25


# 检索函数
def vector_search(query, vectorizer, vectors, df, top_n=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices].copy()
    results['相似度'] = similarities[top_indices]
    return results


def bm25_search(query, bm25, df, texts, top_n=5):
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    results = df.iloc[top_indices].copy()
    results['bm25分数'] = [scores[i] for i in top_indices]
    return results


def search_car(query, vectorizer, vectors, bm25, df, texts, top_n=3):
    query_lower = query.lower()

    # ==================== 1. 动态实体提取 + 多车型对比召回 ====================
    matched_cars = []
    # 动态获取数据库中所有不重复的车型名称
    all_models = df['车型'].dropna().unique()

    import re
    for model in all_models:
        model_str = str(model).strip()
        model_lower = model_str.lower()

        # 策略A：完整匹配（例如用户输入了“红旗H5”或“凯美瑞”）
        if model_lower in query_lower:
            matched_cars.append(model_str)
            continue

        # 策略B：匹配英文字母或数字部分（例如库里是“红旗H5”，用户只说了“H5”）
        eng_part = re.search(r'[a-zA-Z0-9]+', model_lower)
        if eng_part:
            part_str = eng_part.group()
            # 过滤掉单字母或太短的，防止误伤（比如匹配到'5'）
            if len(part_str) >= 2 and part_str in query_lower:
                matched_cars.append(model_str)

    # 去重
    matched_cars = list(set(matched_cars))

    # 如果识别到两个及以上车型，触发对比召回
    if len(matched_cars) >= 2:
        condition = pd.Series([False] * len(df))
        for car in matched_cars:
            # 精确匹配车型名，把所有相关的版本都捞出来
            condition = condition | (df['车型'].str.strip() == car)
        result_df = df[condition].copy()
        if not result_df.empty:
            return result_df, texts, matched_cars

    # ==================== 2. 数值排序兜底 ====================
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
        if keyword in query_lower:
            if field in df.columns:
                full_df = df.copy()
                full_df[field] = pd.to_numeric(full_df[field], errors='coerce')
                full_df = full_df.dropna(subset=[field])
                if not full_df.empty:
                    full_df = full_df.sort_values(by=field, ascending=ascending)
                    result_df = full_df.head(top_n)
                    return result_df, texts, matched_cars

    # ==================== 3. 多路召回 (向量 + BM25) ====================
    vec_results = vector_search(query_lower, vectorizer, vectors, df, top_n=5)
    bm25_results = bm25_search(query_lower, bm25, df, texts, top_n=5)

    merged_indices = []
    for idx in bm25_results.index:
        if idx not in merged_indices:
            merged_indices.append(idx)
    for idx in vec_results.index:
        if idx not in merged_indices:
            merged_indices.append(idx)

    result_df = df.iloc[merged_indices].head(top_n)
    return result_df, texts, matched_cars
def log_user_interaction(session_id, customer_id, user_input, ai_response, retrieved_models, matched_cars):
    log_file = os.path.join(BASE_DIR, "data", "user_logs.csv")
    file_exists = os.path.exists(log_file)
    header = ["timestamp", "session_id", "customer_id", "user_input", "ai_response", "retrieved_models", "matched_cars"]

    # 确保 data 目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(log_file) == 0:
            writer.writerow(header)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session_id,
            customer_id if customer_id else "",
            user_input,
            ai_response,
            ", ".join(retrieved_models) if retrieved_models else "",
            ", ".join(matched_cars) if matched_cars else ""
        ])

def get_customer_conversations(customer_id):
    """
    根据 customer_id 从日志中读取对话记录
    返回: list of dict，按时间正序排列
    """
    import csv
    log_file = "data/user_logs.csv"
    if not os.path.exists(log_file):
        return []

    conversations = []
    target_id = str(customer_id).strip()

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            return []

        # 解析表头
        header = rows[0]
        # 尝试找到各列索引（忽略大小写和空格）
        col_map = {}
        for idx, col_name in enumerate(header):
            col_clean = col_name.strip().lower()
            if 'timestamp' in col_clean:
                col_map['timestamp'] = idx
            elif 'session' in col_clean and 'id' in col_clean:
                col_map['session_id'] = idx
            elif 'customer' in col_clean and 'id' in col_clean:
                col_map['customer_id'] = idx
            elif 'user' in col_clean and 'input' in col_clean:
                col_map['user_input'] = idx
            elif 'ai' in col_clean and 'response' in col_clean:
                col_map['ai_response'] = idx
            elif 'retrieved' in col_clean:
                col_map['retrieved_models'] = idx
            elif 'matched' in col_clean:
                col_map['matched_cars'] = idx

        # 检查必要列是否存在
        if 'customer_id' not in col_map or 'user_input' not in col_map:
            # 如果找不到，尝试按位置推断（常见的列顺序）
            # timestamp, session_id, customer_id, user_input, ai_response, retrieved_models, matched_cars
            if len(header) >= 7:
                col_map['timestamp'] = 0
                col_map['session_id'] = 1
                col_map['customer_id'] = 2
                col_map['user_input'] = 3
                col_map['ai_response'] = 4
                col_map['retrieved_models'] = 5
                col_map['matched_cars'] = 6
            else:
                return []

        # 获取各列索引
        cust_idx = col_map['customer_id']
        user_idx = col_map['user_input']
        ai_idx = col_map.get('ai_response', user_idx + 1)  # 如果没找到，假设在 user_input 后面

        # 遍历数据行
        for row in rows[1:]:
            if len(row) <= max(cust_idx, user_idx, ai_idx):
                continue
            # 去掉可能的前后空格后再比较
            cust_val = row[cust_idx].strip()
            if cust_val == target_id:
                user_content = row[user_idx].strip() if len(row) > user_idx else ""
                ai_content = row[ai_idx].strip() if len(row) > ai_idx else ""
                if user_content:
                    conversations.append({"role": "user", "content": user_content})
                if ai_content:
                    conversations.append({"role": "assistant", "content": ai_content})

        return conversations

    except Exception as e:
        # 如果出错，打印错误并返回空列表
        print(f"读取日志文件出错: {e}")
        return []


def generate_response(query, results_df, texts, history_messages=None):
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

    # 构建对话消息列表
    messages = [
        {"role": "system", "content": """
    你是一名红旗4S店的金牌销售。你专业、真诚，懂得把冷冰冰的参数转化为客户未来的生活画面（场景化营销）。

    【一、绝对红线（违反即失职）】
    1. 严禁推荐非红旗品牌。当客户提到竞品时，绝对不准列举竞品优点（如智驾好、续航长），只能客观对比，并落脚在红旗的优势上（如内饰更豪华、底盘更稳、无续航焦虑）。
    2. 严禁编造具体数值。如果参考数据里没有真实油耗、选装价格，话术标准：“油耗受路况和驾驶习惯影响很大，我无法给您绝对准确的数字。建议您到店试驾，看试驾车的表显数据最真实。”绝对不要引导客户去第三方汽车平台。
    3. 只能基于【参考数据】回答。如果数据为空，请使用兜底话术：“这款车的细节我这边暂时没有完整参数，不过红旗在这个级别一直做得很好，您可以来店里详细看看。”

    【二、话术风格：自然种草，拒绝模板】
    1. 绝对不要使用“专业解答”、“场景描绘”、“第一、第二”这类生硬的小标题！不要分点列出！要写成一段连贯、自然、像面对面聊天一样的话。
    2. 你的回答逻辑应该是：先顺着客户的需求给予肯定，然后自然地抛出参数对比，最后用一两句话，把参数转化为客户未来的生活画面。
    3. 过渡要自然，多用“其实您想想”、“比如周末”、“想象一下”这种口语化的词，不知不觉把客户代入场景。

    【三、场景化话术指南（内化到回答中，不要照抄）】
    * 当聊到空间/轴距：“这车轴距快3米了，您想想，周末带家人去露营，后排装上儿童座椅，老婆还能轻松跷个二郎腿，后备箱塞下推车和帐篷都不费劲，这就是实实在在的从容。”
    * 当聊到动力/操控：“想象一下在高速上超车，您深踩一脚油门，动力随叫随到，干脆利落，底盘稳稳贴地，家里人坐后排也不会觉得晃。”
    * 当聊到配置/豪华：“下班接完老婆，打开氛围灯放首喜欢的歌，在这个私密空间里，一天的疲惫就卸下来了。”

    【四、说话风格】
    - 称呼客户为“您”，绝对禁止使用“兄弟”、“老哥”等江湖气称呼。
    - 保持精简，不要长篇大论，就像面对面聊天一样。
    """}
    ]

    # 添加历史对话（只取最近 6 条，避免上下文过长）
    if history_messages:
        for msg in history_messages[-6:]:
            messages.append(msg)

    # 添加当前用户问题（带参考数据）
    messages.append({"role": "user", "content": f"用户问题：{query}\n\n参考数据：\n{context}"})

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",  # 注意模型名称
            messages=messages,
            stream=True
        )
        return response
    except Exception as e:
        return f"调用API失败：{str(e)}"

# ==================== 销售工作台（新增） ====================

def render_sales_dashboard():
    """销售工作台页面"""
    st.markdown("## 🧑‍💼 销售工作台")

    # 权限校验：只有销售可访问
    if not st.session_state.user or st.session_state.user['role'] != 'sales':
        st.warning("⚠️ 只有销售顾问才能访问此页面")
        return

    current_sales = st.session_state.user['name']
    st.info(f"当前登录：{current_sales}")

    # 获取数据
    my_customers = crm.get_my_customers(current_sales)
    unassigned = crm.get_unassigned_customers()

    # 显示我的客户
    st.markdown("### 👤 我的客户")
    if my_customers:
        for row in my_customers:
            with st.expander(f"{row['name'] or '未命名'}（📱 {row['phone']}）"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**🚗 车型偏好**：{row['preferred_model'] or '待分析'}")
                    st.write(f"**💰 预算范围**：{row['budget_range'] or '待分析'}")
                with col2:
                    st.write(f"**🎯 用车场景**：{row['purpose'] or '待分析'}")
                    st.write(f"**📌 核心诉求**：{row['key_demands'] or '待分析'}")

                # 显示对话回顾
                if row.get('notes'):
                    with st.expander("📋 查看完整对话回顾", expanded=False):
                        st.write(row['notes'])
                else:
                    st.caption("📝 暂无对话回顾")

                # 操作按钮（生成画像 + 移除客户）
                col_btn1, col_btn2 = st.columns(2)

                with col_btn1:
                    if st.button(f"🧠 生成画像", key=f"analyze_{row['id']}"):
                        with st.spinner("正在分析客户对话..."):
                            conversations = get_customer_conversations(row['id'])
                            if conversations:
                                profile = crm.analyze_customer_from_conversations(conversations, client)
                                if profile:
                                    crm.save_profile(row['id'], profile)
                                    st.success("✅ 画像分析完成！")
                                    st.rerun()
                                else:
                                    st.error("❌ 分析失败，请重试")
                            else:
                                st.warning("⚠️ 该客户暂无对话记录，请让客户先与AI聊天")

                # ⭐ 新增：移除客户按钮
                with col_btn2:
                    if st.button(f"🗑️ 移除客户", key=f"remove_{row['id']}"):
                        if crm.release_customer(row['id']):
                            st.warning(f"✅ 已移除客户 {row['phone']}，客户已回到公海池")
                            st.rerun()
                        else:
                            st.error("❌ 移除失败，请重试")
    else:
        st.info("暂无已认领客户")

    st.markdown("---")

    # 显示公海池（未分配客户）
    st.markdown("### 📋 待分配客户")
    if unassigned:
        for row in unassigned:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{row['name'] or '未命名'}**（{row['phone']}）")
            with col2:
                st.caption(f"注册于 {row['registered_at'][:10]}")
            with col3:
                if st.button("认领", key=f"claim_{row['id']}"):
                    if crm.assign_customer_to_sales(row['id'], current_sales):
                        st.success(f"✅ 已认领客户 {row['phone']}")
                        st.rerun()
                    else:
                        st.error("❌ 认领失败，可能已被其他销售认领")
    else:
        st.info("🎉 暂无未分配客户")



# ==================== Streamlit UI ====================
st.set_page_config(page_title="红旗汽车智能销售助手", page_icon="🚗", layout="wide")
# ===== 自定义 CSS + JS（固化手动修改） =====
st.markdown("""
<style>
    /* ===== 1. 极光渐变背景（覆盖所有层级） ===== */
    html, body, .stApp {
        background: 
        radial-gradient(ellipse at 10% 20%, #db8a8a 0%, transparent 55%),
        radial-gradient(ellipse at 90% 80%, rgba(244, 204, 65, 0.33) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 50%, rgba(130, 180, 220, 0.15) 0%, transparent 60%),
        linear-gradient(145deg, #ece7e2 0%, rgba(241, 207, 133, 0.26) 50%, rgba(47, 48, 38, 0.2) 100%) !important;
    background-attachment: fixed;
        margin: 0 !important;
        padding: 0 !important;
        background-color: transparent !important;
    }

    /* ===== 2. 强制干掉所有内联白色背景（包括 JS 设置的） ===== */
    [style*="background: rgb(255, 255, 255)"],
    [style*="background-color: rgb(255, 255, 255)"],
    [style*="background: white"],
    [style*="background-color: white"],
    [style*="background:#fff"],
    [style*="background:#FFFFFF"] {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* ===== 3. 强制所有 Streamlit 容器透明 ===== */
    .stAppViewContainer,
    section[data-testid="stAppViewContainer"],
    .stAppViewContainer > div,
    section.main,
    section.main > div,
    .st-emotion-cache-1r6slb0,
    .st-emotion-cache-18e3th9,
    .st-emotion-cache-16idsys,
    .st-emotion-cache-6px8kg,
    .st-emotion-cache-1j22a0y,
    .st-emotion-cache-scp8yw,
    .egzxvld0 {
        background: transparent !important;
        box-shadow: none !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
    }

    /* ===== 4. 顶部 Header（透明，不遮挡内容） ===== */
    header.stAppHeader,
    .st-emotion-cache-lup3yna,
    .stAppHeader {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        /* 不隐藏 header，保留部署按钮等功能 */
    }

    /* ===== 5. 主区块重置内边距 ===== */
    .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        max-width: 1200px !important;
        background: transparent !important;
    }

    /* ===== 6. 顶部 Banner（渐变红色 + 避开 header） ===== */
    .brand-banner {
        background: linear-gradient(135deg, #8B0000 0%, #cc3344 50%, #8B0000 100%) !important;
        padding: 2rem 2.4rem !important;
        border-radius: 20px;
        margin-top: 3.75rem !important;    /* 重要：避开顶部 header 高度 */
        margin-bottom: 2rem !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        min-height: 90px;
        width: 100%;
        box-sizing: border-box;
    }
    .brand-banner .logo-area {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .brand-banner .logo-icon { font-size: 2rem; }
    .brand-banner .brand-name {
        color: #fff;
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .brand-banner .brand-name span {
        color: #ffd700;
        font-weight: 300;
    }
    .brand-banner .brand-slogan {
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
        font-weight: 300;
    }
    .brand-banner .brand-slogan em {
        color: #ffd700;
        font-style: normal;
    }

    /* ===== 7. 玻璃拟态核心 ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(48px) !important;
        -webkit-backdrop-filter: blur(48px) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 24px;
        box-shadow: 
            0 8px 40px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        padding: 1rem 1.2rem;
        transition: all 0.3s ease;
    }

    /* ===== 8. 侧边栏 ===== */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(56px) !important;
        -webkit-backdrop-filter: blur(56px) !important;
        border-right: 1.5px solid rgba(255, 255, 255, 0.4) !important;
        padding: 1.2rem 0.8rem !important;
        margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] .sidebar-brand {
        text-align: center;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] .sidebar-brand .brand-icon {
        font-size: 1.8rem;
    }
    section[data-testid="stSidebar"] .sidebar-brand .brand-text {
        font-size: 1rem;
        font-weight: 600;
        color: #3a1a1a;
        text-shadow: 0 1px 4px rgba(255,255,255,0.3);
    }

    /* ===== 9. “大家都在问” 按钮颜色优化 ===== */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 14px;
        padding: 0.5rem 0.8rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #2c1a1a !important;
        width: 100%;
        text-align: left;
        transition: all 0.2s ease;
        box-shadow: none;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.4) !important;
        border-color: #7a1a1a !important;
        color: #7a1a1a !important;
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] h3 {
        color: #4a1a1a !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.3px;
    }

    /* ===== 10. 消息气泡 ===== */
    .user-message {
        background: rgba(144, 0, 26, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        color: #edb974 !important; 
        padding: 0.8rem 1.4rem;
        border-radius: 24px 24px 6px 24px;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 4px 24px rgba(122, 26, 26, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        font-size: 0.95rem;
        line-height: 1.6;
        animation: slideInRight 0.25s ease;
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(32px) !important;
        -webkit-backdrop-filter: blur(32px) !important;
        color: #1a1a1a;
        padding: 0.8rem 1.4rem;
        border-radius: 24px 24px 24px 6px;
        max-width: 75%;
        box-shadow: 0 4px 32px rgba(0, 0, 0, 0.03);
        border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
        font-size: 0.95rem;
        line-height: 1.7;
        animation: slideInLeft 0.25s ease;

    }
    .assistant-message h1,
    .assistant-message h2,
    .assistant-message h3,
    .assistant-message h4,
    .assistant-message h5,
    .assistant-message h6 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin: 0.5rem 0 !important;
    }
    .assistant-message ul,
    .assistant-message ol {
        padding-left: 1.5rem !important;
    }
    .assistant-message li {
        margin-bottom: 0.2rem !important;
    }
    

    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(16px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-16px); }
        to { opacity: 1; transform: translateX(0); }
    }


    .stChatInput > div {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 40px !important;
        padding: 0.2rem 0.4rem !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.02);
    }
    .stChatInput > div:focus-within {
        border-color: #7a1a1a !important;
        background: rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 0 0 4px rgba(122, 26, 26, 0.06);
    }
    .stChatInput input {
        padding: 0.6rem 1.2rem !important;
        font-size: 0.95rem !important;
    }

    /* ===== 12. 其他组件 ===== */
    .stButton > button {
        background: #7a1a1a !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.3rem 1.6rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #4a0e0e !important;
        transform: scale(1.02) !important;
        box-shadow: 0 4px 12px rgba(122, 26, 26, 0.2) !important;
    }

    .stExpander {
        background: transparent !important;
        border: none !important;
    }
    .stExpander .streamlit-expanderHeader {
        background: rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 0.2rem 0.8rem !important;
        color: #444 !important;
    }
    
    /* 强制隐藏密码框内部所有按钮 */
    .stTextInput div button {
        display: none !important;
    }
    
    /* 隐藏密码框右侧的切换按钮容器 */
    .stTextInput .st-emotion-cache-1kyxreq {
       display: none !important;
    }



    /* ===== 14. 响应式 ===== */
    @media (max-width: 768px) {
        .brand-banner {
            flex-direction: column;
            text-align: center;
            padding: 1.2rem !important;
            margin-top: 3.5rem !important;  /* 移动端 header 高度可能不同 */
        }
        .brand-banner .brand-slogan {
            margin-top: 0.3rem;
        }
        .user-message, .assistant-message {
            max-width: 90%;
        }
    }

    div[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
div[data-testid="stBottom"] * {
    background: transparent !important;
    background-color: transparent !important;
}/* ===== 覆盖 st.chat_message 默认样式，使其与自定义样式一致 ===== */
/* 用户消息（发送瞬间） */
[data-testid="chatMessage"] {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
/* 用户消息：覆盖内部容器 */
[data-testid="chatMessage"][data-role="user"] .st-emotion-cache-1kyxreq,
[data-testid="chatMessage"][data-role="user"] .st-emotion-cache-1w3d5e1 {
    background: rgba(144, 0, 32, 0.8) !important;
    color: #edb974 !important;
    border-radius: 24px 24px 6px 24px !important;
    padding: 0.8rem 1.4rem !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 4px 24px rgba(122, 26, 26, 0.15) !important;
}
/* AI消息：覆盖内部容器 */
[data-testid="chatMessage"][data-role="assistant"] .st-emotion-cache-1kyxreq,
[data-testid="chatMessage"][data-role="assistant"] .st-emotion-cache-1w3d5e1 {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(32px) !important;
    -webkit-backdrop-filter: blur(32px) !important;
    color: #1a1a1a !important;
    border-radius: 24px 24px 24px 6px !important;
    padding: 0.8rem 1.4rem !important;
    border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.03) !important;
}
/* 去掉默认头像的干扰 */
[data-testid="chatMessage"] .st-emotion-cache-1w3d5e1 img,
[data-testid="chatMessage"] .st-emotion-cache-1w3d5e1 svg {
    display: none !important;
}
[data-testid="chatMessage"] .st-emotion-cache-1kyxreq {
    gap: 0 !important;
}

/* ===== 隐藏 Streamlit 默认密码框的眼睛图标 ===== */
/* 方法1：通用隐藏 */
.stTextInput button {
    display: none !important;
}

/* 方法2：精准定位（如果方法1不生效） */
div[data-testid="stTextInput"] button {
    display: none !important;
}

/* 方法3：隐藏整个按钮容器 */
.stTextInput .st-emotion-cache-1kyxreq {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ===== 加载数据 =====
df = load_car_data()
if df is None:
    st.stop()

# ===== 构建索引 =====
texts = build_texts(df)
vectorizer, vectors, _ = build_vector_index(df)
bm25 = build_bm25_index(df, texts)


# ===== 初始化热门问题管理器（使用全局状态，跨用户同步） =====
# 使用官方全局缓存，所有用户共享同一个实例
@st.cache_resource
def get_global_hot_mgr():
    return HotQuestionsManager(storage_file=os.path.join(BASE_DIR, "data", "hot_questions.json"))

hot_mgr = get_global_hot_mgr()



st.markdown("""
<div class="brand-banner">
    <div class="logo-area">
        <div class="logo-icon">🚗</div>
        <div class="brand-name">红旗<span>汽车</span></div>
    </div>
    <div class="brand-slogan">让理想飞扬 · <em>AI 智能选车顾问</em></div>
</div>
""", unsafe_allow_html=True)

# ===== 侧边栏 =====
# ===== 侧边栏 =====
with st.sidebar:
    # ----- 登录/注册模块（无 form，交互实时响应） -----
    if 'user' not in st.session_state:
        st.session_state.user = None

    # 已登录状态
    if st.session_state.user is not None:
        user = st.session_state.user
        if user['role'] == 'customer':
            st.write(f"👤 **{user['name']}**（客户）")
        else:
            st.write(f"👤 **{user['name']}**（销售顾问）")

        # 员工专属入口
        if user['role'] == 'sales':
            if st.button("📋 销售工作台", key="goto_sales_dashboard", use_container_width=True):
                st.session_state.show_sales_dashboard = True
                st.rerun()

        if st.button("🚪 退出登录", key="logout_btn", use_container_width=True):
            st.session_state.user = None
            st.session_state.show_sales_dashboard = False
            # 重置聊天记录
            st.session_state.messages = [
                {"role": "assistant", "content": "你好，我是红旗 AI 销售顾问。请问你想了解哪款红旗车型？"}]
            st.session_state.quick_question = None
            st.session_state.pending_question = None
            st.rerun()
        st.markdown("---")
    else:
        # 未登录：显示登录/注册弹窗
        with st.popover("🔑 登录 / 注册", use_container_width=True):
            # 定义两个 Tab
            tab_login, tab_register = st.tabs(["🔑 登录", "📝 注册"])

            with tab_login:
                # 不用 st.form，直接放组件
                role_login = st.radio("角色", ["客户", "员工"], key="login_role", horizontal=True)
                phone_login = st.text_input("手机号", key="login_phone_modal")
                # 复选框控制密码是否明文显示（勾选后立即 rerun，生效）
                show_pwd_login = st.checkbox("显示密码", key="show_pwd_login")
                if show_pwd_login:
                    password_login = st.text_input("密码（明文）", key="login_password_modal")
                else:
                    password_login = st.text_input("密码", type="password", key="login_password_modal")
                # 登录按钮（普通按钮，点击后执行逻辑）
                if st.button("登录", key="login_btn_modal"):
                    if role_login == "客户":
                        user = crm.login_customer(phone_login, password_login)
                        if user:
                            st.session_state.user = user
                            # 重置聊天记录
                            st.session_state.messages = [
                                {"role": "assistant", "content": "你好，我是红旗 AI 销售顾问。请问你想了解哪款红旗车型？"}]
                            st.session_state.quick_question = None
                            st.session_state.pending_question = None
                            st.success("登录成功！")
                            st.rerun()
                        else:
                            st.error("手机号或密码错误")
                    else:
                        user = crm.login_sales_rep(phone_login, password_login)
                        if user:
                            st.session_state.user = user
                            # ⭐ 新增：重置聊天记录
                            st.session_state.messages = [
                                {"role": "assistant", "content": "你好，我是红旗 AI 销售顾问。请问你想了解哪款红旗车型？"}]
                            st.session_state.quick_question = None
                            st.session_state.pending_question = None
                            st.success("登录成功！")
                            st.rerun()
                        else:
                            st.error("手机号或密码错误")

            with tab_register:
                role_reg = st.radio("注册角色", ["客户", "员工"], key="reg_role_modal", horizontal=True)
                phone_reg = st.text_input("手机号", key="reg_phone_modal")
                name_reg = st.text_input("姓名", key="reg_name_modal")
                show_pwd_reg = st.checkbox("显示密码", key="show_pwd_reg")
                if show_pwd_reg:
                    password_reg = st.text_input("密码（明文）", key="reg_password_modal")
                    confirm_reg = st.text_input("确认密码（明文）", key="reg_confirm_modal")
                else:
                    password_reg = st.text_input("密码", type="password", key="reg_password_modal")
                    confirm_reg = st.text_input("确认密码", type="password", key="reg_confirm_modal")
                if st.button("注册", key="reg_btn_modal"):
                    if password_reg != confirm_reg:
                        st.error("两次密码不一致")
                    elif len(phone_reg) < 11:
                        st.error("请输入有效的手机号")
                    else:
                        if role_reg == "客户":
                            result = crm.register_customer(phone_reg, name_reg, password_reg)
                            if result:
                                st.session_state.user = result
                                st.session_state.messages = [{"role": "assistant",
                                                              "content": "你好，我是红旗 AI 销售顾问。请问你想了解哪款红旗车型？"}]
                                st.session_state.quick_question = None
                                st.session_state.pending_question = None
                                st.success("注册成功，已自动登录！")
                                st.rerun()
                            else:
                                st.error("手机号已被注册")
                        else:
                            result = crm.register_sales_rep(phone_reg, name_reg, password_reg)
                            if result:
                                st.session_state.user = result
                                # ⭐ 新增：重置聊天记录
                                st.session_state.messages = [{"role": "assistant",
                                                             "content": "你好，我是红旗 AI 销售顾问。请问你想了解哪款红旗车型？"}]
                                st.session_state.quick_question = None
                                st.session_state.pending_question = None
                                st.success("注册成功，已自动登录！")
                                st.rerun()
                            else:
                                st.error("手机号或姓名已被注册")
        st.markdown("---")

    # ----- 原有的侧边栏内容（系统状态、热门问题等）-----
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🚗</div>
        <div class="brand-text">红旗 · AI 助手</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 系统状态", expanded=True):
        hongqi_count = len(df[df['品牌'] == '红旗']) if '品牌' in df.columns else len(df)
        st.success(f"✅ 已加载 {len(df)} 款车型（红旗 {hongqi_count} 款）")
        st.info("🧠 引擎: TF-IDF + BM25")

    st.markdown("---")
    st.markdown("### 🔥 大家都在问")

    top_questions = hot_mgr.get_top_questions(5)
    if len(top_questions) < 3:
        defaults = ["红旗H9多少钱", "哪款车适合二胎家庭", "红旗H9和奥迪A6L哪个好"]
        for q in defaults:
            if q not in top_questions:
                top_questions.append(q)
        top_questions = top_questions[:5]

    for q in top_questions:
        if st.button(f"{q}", key=f"hot_{q}"):
            st.session_state.quick_question = q

    st.markdown("---")
    with st.expander("💡 使用提示", expanded=False):
        st.markdown("""
        - 输入车型名称查询详细参数
        - 问 **最贵/最便宜** 进行价格排序
        - 输入两个车型（如 **H9 vs H5**）进行对比
        """)

# ===== 初始化 session_state =====
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是红旗 AI 销售顾问。请问你想了解哪款红旗车型？"}
    ]
if "quick_question" not in st.session_state:
    st.session_state.quick_question = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
# ===== 新增：销售工作台开关 =====
if "show_sales_dashboard" not in st.session_state:
    st.session_state.show_sales_dashboard = False

# ===== 处理热门问题点击 =====
if st.session_state.get("quick_question"):
    st.session_state.pending_question = st.session_state.quick_question
    st.session_state.quick_question = None

# ===== 页面路由 =====
if st.session_state.get("show_sales_dashboard", False):
    # ----- 销售工作台页面 -----
    render_sales_dashboard()
    # 返回按钮
    if st.button("← 返回智能选车", key="back_to_chat"):
        st.session_state.show_sales_dashboard = False
        st.rerun()
else:
    # ===== 智能选车聊天界面 =====
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-end; margin: 0.6rem 0;"><div class="user-message">{msg["content"]}</div></div>',
                    unsafe_allow_html=True)
            else:
                html_content = markdown.markdown(msg["content"])
                st.markdown(
                    f'<div style="display: flex; justify-content: flex-start; margin: 0.6rem 0;"><div class="assistant-message">{html_content}</div></div>',
                    unsafe_allow_html=True)
    # 输入框：先定义 user_input = None，确保变量始终存在
    user_input = None

    if st.session_state.get("pending_question"):
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        user_input = st.chat_input("💬 请输入你关心的问题...")

    # 只有在 user_input 有内容时才处理
    if user_input is not None and user_input != "":

        # 记录问题
        hot_mgr.record_question(user_input, client=client)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end; margin: 0.6rem 0;"><div class="user-message">{user_input}</div></div>',
            unsafe_allow_html=True
        )

        # ----- 检索（带加载动画） -----
        with st.spinner("🔍 正在检索车型..."):
            results_df, _, matched_cars = search_car(user_input, vectorizer, vectors, bm25, df, texts)
        st.session_state.last_results = results_df


        # ----- 生成回答（流式） -----
        response_stream = generate_response(
            user_input,
            results_df,
            texts,
            history_messages=st.session_state.messages
        )

        #加载动画： 占位符用于流式输出，初始显示加载提示（带转圈动画）
        placeholder = st.empty()
        placeholder.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin: 0.6rem 0;">
                <svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <style>
                        @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                        }
                        .spinner { animation: spin 1s linear infinite; transform-origin: center; }
                    </style>
                    <circle cx="12" cy="12" r="10" fill="none" stroke="#8B0000" stroke-width="3" opacity="0.2"/>
                    <path class="spinner" d="M12 2 A10 10 0 0 1 22 12" fill="none" stroke="#8B0000" stroke-width="3" stroke-linecap="round"/>
                </svg>
                <span style="font-size: 1rem; color: #555;">正在生成回答...</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        full_response = ""
        first_chunk = True

        if hasattr(response_stream, '__iter__') and not isinstance(response_stream, str):
            for chunk in response_stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        if first_chunk:
                            # 第一次收到内容，清空加载提示
                            placeholder.empty()
                            first_chunk = False
                        full_response += delta.content
                        # 每收到一块内容，用 markdown 转换后刷新显示
                        html_content = markdown.markdown(full_response)
                        placeholder.markdown(
                            f'<div style="display: flex; justify-content: flex-start; margin: 0.6rem 0;"><div class="assistant-message">{html_content}</div></div>',
                            unsafe_allow_html=True
                        )
                else:
                    # 异常情况兼容
                    if first_chunk:
                        placeholder.empty()
                        first_chunk = False
                    full_response += str(chunk)
                    html_content = markdown.markdown(full_response)
                    placeholder.markdown(
                        f'<div style="display: flex; justify-content: flex-start; margin: 0.6rem 0;"><div class="assistant-message">{html_content}</div></div>',
                        unsafe_allow_html=True
                    )
        else:
            # 非流式响应
            full_response = response_stream
            html_content = markdown.markdown(full_response)
            placeholder.markdown(
                f'<div style="display: flex; justify-content: flex-start; margin: 0.6rem 0;"><div class="assistant-message">{html_content}</div></div>',
                unsafe_allow_html=True
            )

        # 保存完整回答（存入 session_state 时不转换，保留原始 Markdown）
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # 7. 记录日志
        retrieved_models = results_df['车型'].tolist() if not results_df.empty else []
        customer_id = None
        if st.session_state.user and st.session_state.user['role'] == 'customer':
            customer_id = st.session_state.user.get('id')
        log_user_interaction(
            session_id=st.session_state.get("session_id", "unknown"),
            customer_id=customer_id,
            user_input=user_input,
            ai_response=full_response,
            retrieved_models=retrieved_models,
            matched_cars=matched_cars
        )

        st.rerun()