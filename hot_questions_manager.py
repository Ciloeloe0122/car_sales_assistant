import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import re
import jieba


class HotQuestionsManager:
    def __init__(self, storage_file="data/hot_questions.json"):
        self.storage_file = storage_file
        self.data = self._load()
        # 车型关键词映射（把简称统一成标准名）
        self.car_aliases = {
            "h9": "红旗H9",
            "h5": "红旗H5",
            "hs5": "红旗HS5",
            "hs7": "红旗HS7",
            "h6": "红旗H6",
            "eqm5": "红旗E-QM5",
            "ehs9": "红旗E-HS9",
            "hs3": "红旗HS3",
            "hq9": "红旗HQ9",
        }
        # 意图关键词
        self.intent_keywords = {
            "价格": ["多少", "价格", "报价", "价位", "多少钱", "落地", "优惠"],
            "配置": ["配置", "有什么", "什么功能", "带什么", "有没有", "参数", "尺寸", "轴距", "空间", "动力", "加速",
                     "续航"],
            "对比": ["和", "对比", "比较", "哪个好", "怎么选", "区别"],
            "推荐": ["推荐", "适合", "哪款", "选哪", "建议"],
        }

    def _load(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"questions": {}, "daily": {}}
        return {"questions": {}, "daily": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _normalize_question(self, question):
        """把问题归一化成标准键"""
        clean = question.strip().rstrip("？?").strip()
        clean_lower = clean.lower()

        # 1. 识别车型
        car = None
        # 先找完整名称
        for alias in self.car_aliases.keys():
            if alias in clean_lower or self.car_aliases[alias] in clean:
                car = self.car_aliases[alias]
                break
        # 如果没找到，用 jieba 分词提取
        if not car:
            words = list(jieba.cut(clean))
            for word in words:
                if "红旗" in word or "H" in word or "HS" in word or "EQ" in word:
                    for alias, full in self.car_aliases.items():
                        if alias.upper() in word.upper() or full in word:
                            car = full
                            break
                    if car:
                        break

        # 2. 识别意图
        intent = "通用"
        for intent_name, keywords in self.intent_keywords.items():
            for kw in keywords:
                if kw in clean:
                    intent = intent_name
                    break
            if intent != "通用":
                break

        # 3. 如果没识别到车型，用问题本身作为键（但不推荐）
        if not car:
            # 去掉最常见的问法，保留核心
            clean_core = re.sub(r'多少钱|价格|报价|配置|推荐|适合|怎么样|好不好', '', clean)
            if len(clean_core) > 2:
                return f"其他_问题_{clean_core[:10]}"
            return f"其他_问题_{clean[:10]}"

        # 4. 构造归一化键
        return f"{car}_{intent}"

    def _get_best_display_text(self, question, normalized_key):
        """从同组问题中选一个展示文本"""
        # 优先选包含车型全称且长度适中的
        candidates = self.data["questions"].get(normalized_key, {}).get("variants", [])
        if not candidates:
            return question

        # 按长度排序，选中等长度的（太短太模糊，太长太啰嗦）
        candidates.sort(key=len)
        mid = len(candidates) // 2
        return candidates[mid]

    def record_question(self, question):
        """记录一个问题"""
        clean = question.strip().rstrip("？?").strip()
        if not clean:
            return

        # 归一化
        normalized_key = self._normalize_question(clean)

        # 记录到聚合键下
        if normalized_key not in self.data["questions"]:
            self.data["questions"][normalized_key] = {
                "count": 0,
                "variants": [],
                "latest": clean
            }

        # 更新计数（不重复计数同一问题）
        self.data["questions"][normalized_key]["count"] += 1
        if clean not in self.data["questions"][normalized_key]["variants"]:
            self.data["questions"][normalized_key]["variants"].append(clean)
        self.data["questions"][normalized_key]["latest"] = clean

        # 每日记录
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.data["daily"]:
            self.data["daily"][today] = {}
        if normalized_key not in self.data["daily"][today]:
            self.data["daily"][today][normalized_key] = {"count": 0, "variants": []}
        self.data["daily"][today][normalized_key]["count"] += 1
        if clean not in self.data["daily"][today][normalized_key]["variants"]:
            self.data["daily"][today][normalized_key]["variants"].append(clean)

        self._clean_old_data()
        self._save()

    def _clean_old_data(self):
        """清理30天前数据"""
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        to_delete = [d for d in self.data["daily"].keys() if d < cutoff]
        for d in to_delete:
            del self.data["daily"][d]

        # 重新聚合
        total = defaultdict(lambda: {"count": 0, "variants": [], "latest": ""})
        for day, items in self.data["daily"].items():
            for key, info in items.items():
                total[key]["count"] += info["count"]
                total[key]["variants"].extend(info.get("variants", []))
                if info.get("latest"):
                    total[key]["latest"] = info["latest"]

        for key in total:
            total[key]["variants"] = list(set(total[key]["variants"]))
        self.data["questions"] = dict(total)

    def get_top_questions(self, n=5):
        sorted_items = sorted(
            self.data["questions"].items(),
            key=lambda x: x[1]["count"] if isinstance(x[1], dict) else x[1],
            reverse=True
        )
        result = []
        seen = set()
        for key, info in sorted_items:
            if isinstance(info, dict):
                count = info.get("count", 0)
                variants = info.get("variants", [])
            else:
                count = info
                variants = [key]

            display = variants[0] if variants else key
            for v in variants:
                if len(v) > 4 and ("红旗" in v or "H" in v):
                    display = v
                    break

            if display in seen:
                continue
            seen.add(display)

            result.append(f"{display}（热度：{count}）")
            if len(result) >= n:
                break
        return result
    def get_hot_score(self, question):
        """获取某个问题的热度分数"""
        # 先尝试精确匹配，再尝试归一化匹配
        normalized_key = self._normalize_question(question)
        info = self.data["questions"].get(normalized_key)
        if info:
            return info["count"]
        return 0