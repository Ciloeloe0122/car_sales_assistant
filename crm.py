"""
CRM 模块 - 客户关系管理 + 用户认证
包含：数据库初始化、客户/销售注册、登录、客户管理、画像分析
"""
import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from openai import OpenAI


class CRM:
    """CRM 核心类，封装所有数据库操作和业务逻辑"""

    def __init__(self, db_path: str = "crm.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _hash_password(self, password: str) -> str:
        """SHA256 加密密码"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _init_db(self) -> None:
        """初始化数据库表结构（自动修复损坏文件）"""
        import os

        # 1. 检查文件是否存在且有效
        def is_valid_db(path):
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) == 0:
                return False
            try:
                conn = sqlite3.connect(path)
                conn.execute("SELECT 1")
                conn.close()
                return True
            except:
                return False

        # 2. 如果文件存在但无效，删除它
        if os.path.exists(self.db_path) and not is_valid_db(self.db_path):
            os.remove(self.db_path)

        # 3. 连接数据库并建表
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建客户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(50),
                password_hash VARCHAR(255) NOT NULL,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建销售顾问表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales_rep (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建客户归属表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_assignment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                sales_rep_id INTEGER NOT NULL,
                assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active',
                FOREIGN KEY(customer_id) REFERENCES customer(id),
                FOREIGN KEY(sales_rep_id) REFERENCES sales_rep(id)
            )
        ''')

        # 创建客户画像表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_profile (
                customer_id INTEGER PRIMARY KEY,
                preferred_model VARCHAR(100),
                budget_range VARCHAR(50),
                purpose VARCHAR(50),
                key_demands VARCHAR(200),
                notes TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(customer_id) REFERENCES customer(id)
            )
        ''')

        # 索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_phone ON customer(phone)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assignment_customer ON customer_assignment(customer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assignment_sales_rep ON customer_assignment(sales_rep_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_assignment_status ON customer_assignment(status)")

        conn.commit()
        conn.close()

    # ==================== 用户认证 ====================

    def register_customer(self, phone: str, name: str, password: str) -> Optional[Dict[str, Any]]:
        """
        注册新客户
        :return: 客户信息字典 {'id', 'name', 'role'} 或 None（失败）
        """
        if self.get_customer_by_phone(phone):
            return None
        hash_pw = self._hash_password(password)
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO customer (phone, name, password_hash) VALUES (?, ?, ?)",
                (phone, name, hash_pw)
            )
            conn.commit()
            customer_id = cursor.lastrowid
            conn.close()
            return {'id': customer_id, 'name': name, 'role': 'customer'}
        except sqlite3.IntegrityError:
            conn.close()
            return None

    def register_sales_rep(self, phone: str, name: str, password: str) -> Optional[Dict[str, Any]]:
        """
        注册销售顾问
        :return: 销售信息字典 {'id', 'name', 'role'} 或 None（失败）
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        # 检查手机号或姓名是否已存在
        cursor.execute("SELECT id FROM sales_rep WHERE phone = ? OR name = ?", (phone, name))
        if cursor.fetchone():
            conn.close()
            return None
        hash_pw = self._hash_password(password)
        cursor.execute(
            "INSERT INTO sales_rep (phone, name, password_hash) VALUES (?, ?, ?)",
            (phone, name, hash_pw)
        )
        conn.commit()
        rep_id = cursor.lastrowid
        conn.close()
        return {'id': rep_id, 'name': name, 'role': 'sales'}

    def login_customer(self, phone: str, password: str) -> Optional[Dict[str, Any]]:
        """客户登录，返回用户信息或 None"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name FROM customer WHERE phone = ? AND password_hash = ?",
            (phone, self._hash_password(password))
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'id': row['id'], 'name': row['name'], 'role': 'customer'}
        return None

    def login_sales_rep(self, phone: str, password: str) -> Optional[Dict[str, Any]]:
        """销售登录，返回用户信息或 None"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name FROM sales_rep WHERE phone = ? AND password_hash = ?",
            (phone, self._hash_password(password))
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'id': row['id'], 'name': row['name'], 'role': 'sales'}
        return None

    # ==================== 客户信息查询 ====================

    def get_customer_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """根据手机号查询客户"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customer WHERE phone = ?", (phone,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_customer_by_id(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """根据客户ID查询"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customer WHERE id = ?", (customer_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_sales_rep_by_id(self, rep_id: int) -> Optional[Dict[str, Any]]:
        """根据销售ID查询"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sales_rep WHERE id = ?", (rep_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # ==================== 销售认领功能 ====================

    def assign_customer_to_sales(self, customer_id: int, sales_rep_name: str) -> bool:
        """将客户分配给销售顾问"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM sales_rep WHERE name = ?", (sales_rep_name,))
        rep = cursor.fetchone()
        if not rep:
            conn.close()
            return False
        rep_id = rep['id']

        cursor.execute(
            "SELECT sales_rep_id FROM customer_assignment WHERE customer_id = ? AND status = 'active'",
            (customer_id,)
        )
        existing = cursor.fetchone()
        if existing:
            if existing['sales_rep_id'] == rep_id:
                conn.close()
                return True
            else:
                conn.close()
                return False

        cursor.execute(
            "INSERT INTO customer_assignment (customer_id, sales_rep_id) VALUES (?, ?)",
            (customer_id, rep_id)
        )
        conn.commit()
        conn.close()
        return True

    def release_customer(self, customer_id: int) -> bool:
        """释放客户（状态改为 released）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE customer_assignment SET status = 'released' WHERE customer_id = ? AND status = 'active'",
            (customer_id,)
        )
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0

    def get_my_customers(self, sales_rep_name: str) -> List[Dict[str, Any]]:
        """获取当前销售认领的所有活跃客户（含画像）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                c.id,
                c.phone,
                c.name,
                c.registered_at,
                ca.assigned_at,
                cp.preferred_model,
                cp.budget_range,
                cp.purpose,
                cp.key_demands,
                cp.notes
            FROM customer c
            JOIN customer_assignment ca ON c.id = ca.customer_id
            JOIN sales_rep sr ON ca.sales_rep_id = sr.id
            LEFT JOIN customer_profile cp ON c.id = cp.customer_id
            WHERE sr.name = ? AND ca.status = 'active'
            ORDER BY ca.assigned_at DESC
        ''', (sales_rep_name,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_unassigned_customers(self) -> List[Dict[str, Any]]:
        """获取所有未分配客户（公海池）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.id, c.phone, c.name, c.registered_at
            FROM customer c
            LEFT JOIN customer_assignment ca ON c.id = ca.customer_id AND ca.status = 'active'
            WHERE ca.id IS NULL
            ORDER BY c.registered_at ASC
        ''')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_all_sales_reps(self) -> List[str]:
        """获取所有销售顾问姓名列表"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sales_rep ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return [row['name'] for row in rows]

    # ==================== 客户画像 ====================

    def save_profile(self, customer_id: int, profile_data: Dict[str, Any]) -> bool:
        """保存或更新客户画像"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # ⭐ 关键修复：把 conversation_summary 存入 notes
        notes = profile_data.get('notes')
        if not notes:
            notes = profile_data.get('conversation_summary', '')

        cursor.execute('''
            INSERT OR REPLACE INTO customer_profile
            (customer_id, preferred_model, budget_range, purpose, key_demands, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            customer_id,
            profile_data.get('preferred_model'),
            profile_data.get('budget_range'),
            profile_data.get('purpose'),
            profile_data.get('key_demands'),
            notes,  # 这里用处理后的 notes
        ))
        conn.commit()
        conn.close()
        return True

    def get_profile(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """获取客户画像"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customer_profile WHERE customer_id = ?", (customer_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # ==================== 智能画像分析（调用 DeepSeek） ====================
    def analyze_customer_from_conversations(self, conversations, openai_client):
        """
        基于客户历史对话，使用 DeepSeek 提取结构化画像
        :param conversations: list of dict，每个 dict 有 role 和 content
        :param openai_client: OpenAI 客户端实例
        :return: 包含画像信息的字典
        """
        if not conversations:
            return None

        chat_text = "\n".join([
            f"{'客户' if msg['role'] == 'user' else '销售顾问'}: {msg['content']}"
            for msg in conversations[-30:]  # 扩展到最近30条
        ])

        prompt = f"""
你是一位资深的汽车销售数据分析师。请分析以下客户与销售顾问的对话记录，提取该客户的购车画像。

对话记录：
{chat_text}

请以 JSON 格式返回以下信息（只返回 JSON，不要包含其他文字）：
{{
    "preferred_model": "客户最感兴趣的车型，如果没有明确填'未知'",
    "budget_range": "客户提到的预算范围，如'20-30万'，如果没提到填'未提及'",
    "purpose": "用车场景：commuting(通勤)/business(商务)/family(家庭)/mixed(混合)，如果没明确填'未知'",
    "key_demands": "客户最关注的3个方面，用逗号分隔，如'空间大,安全性高,省油'",
    "conversation_summary": "用一段话（200-300字）完整回顾客户的购车对话过程，包括：客户提出了哪些问题、重点关注哪些车型或配置、表达过哪些顾虑和疑虑、对哪些功能表现出明显兴趣、最终是否明确了购买意向或倾向。要求条理清晰，让销售看完就能全面了解客户情况，便于后续跟进。"
}}
"""

        try:
            response = openai_client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            import json
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"画像分析失败: {e}")
            return None


# ==================== 单例模式 ====================
_crm_instance = None

def get_crm() -> CRM:
    global _crm_instance
    if _crm_instance is None:
        try:
            _crm_instance = CRM()
        except Exception as e:
            print(f"第一次初始化失败，删除数据库文件后重试: {e}")
            # 删除可能损坏的文件
            db_path = "crm.db"
            if os.path.exists(db_path):
                os.remove(db_path)
            _crm_instance = CRM()  # 重新尝试
    return _crm_instance