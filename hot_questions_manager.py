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
        # ===== 红旗车型映射（覆盖全称、简称、用户常见叫法） =====
        self.car_aliases = {
            # ---------- 轿车系列 ----------
            "h9": "红旗H9",
            "红旗h9": "红旗H9",
            "红旗H9": "红旗H9",
            "h9轿车": "红旗H9",
            "红旗旗舰轿车": "红旗H9",

            "h5": "红旗H5",
            "红旗h5": "红旗H5",
            "红旗H5": "红旗H5",
            "h5轿车": "红旗H5",
            "新h5": "红旗H5",
            "红旗b级车": "红旗H5",

            "h6": "红旗H6",
            "红旗h6": "红旗H6",
            "红旗H6": "红旗H6",
            "h6轿跑": "红旗H6",
            "红旗轿跑": "红旗H6",

            "h7": "红旗H7",
            "红旗h7": "红旗H7",
            "红旗H7": "红旗H7",
            "h7轿车": "红旗H7",

            # ---------- SUV系列 ----------
            "hs3": "红旗HS3",
            "红旗hs3": "红旗HS3",
            "红旗HS3": "红旗HS3",
            "hs3suv": "红旗HS3",
            "红旗紧凑型suv": "红旗HS3",
            "红旗小suv": "红旗HS3",

            "hs5": "红旗HS5",
            "红旗hs5": "红旗HS5",
            "红旗HS5": "红旗HS5",
            "hs5suv": "红旗HS5",
            "红旗中型suv": "红旗HS5",
            "红旗suv": "红旗HS5",  # 用户只说"红旗suv"时默认HS5

            "hs6": "红旗HS6",
            "红旗hs6": "红旗HS6",
            "红旗HS6": "红旗HS6",

            "hs7": "红旗HS7",
            "红旗hs7": "红旗HS7",
            "红旗HS7": "红旗HS7",
            "hs7suv": "红旗HS7",
            "红旗中大型suv": "红旗HS7",

            "hs9": "红旗HS9",
            "红旗hs9": "红旗HS9",
            "红旗HS9": "红旗HS9",
            "红旗全尺寸suv": "红旗HS9",

            # ---------- MPV系列 ----------
            "hq9": "红旗HQ9",
            "红旗hq9": "红旗HQ9",
            "红旗HQ9": "红旗HQ9",
            "红旗mpv": "红旗HQ9",
            "红旗商务车": "红旗HQ9",

            # ---------- 新能源系列 ----------
            "eqm5": "红旗E-QM5",
            "红旗eqm5": "红旗E-QM5",
            "红旗E-QM5": "红旗E-QM5",
            "eqm5纯电": "红旗E-QM5",
            "红旗纯电轿车": "红旗E-QM5",

            "eh7": "红旗EH7",
            "红旗eh7": "红旗EH7",
            "红旗EH7": "红旗EH7",
            "eh7纯电": "红旗EH7",

            "ehs9": "红旗E-HS9",
            "红旗ehs9": "红旗E-HS9",
            "红旗E-HS9": "红旗E-HS9",
            "ehs9纯电": "红旗E-HS9",
            "红旗纯电suv": "红旗E-HS9",
            "红旗大型纯电suv": "红旗E-HS9",

            "红旗新能源": "红旗新能源",
            "红旗电动车": "红旗新能源",
            "红旗纯电车": "红旗新能源",
            "红旗电车": "红旗新能源",

            # ---------- 品牌通用 ----------
            "红旗": "红旗",  # 只提品牌不提具体车型
        }

        # ===== 意图关键词（按用户真实提问习惯分类） =====
        self.intent_keywords = {
            # 价格/预算类
            "价格": [
                "多少钱", "价格", "报价", "价位", "落地价", "落地", "优惠", "便宜",
                "贵不贵", "性价比", "划算", "预算", "值不值", "有优惠", "折扣",
                "最低价", "成交价", "指导价", "裸车价", "全款", "分期", "贷款",
                "月供", "首付", "利息", "金融", "免息"
            ],

            # 配置/功能类
            "配置": [
                "配置", "有什么", "什么功能", "带什么", "有没有", "标配", "高配",
                "低配", "顶配", "选装", "加装", "功能", "参数", "尺寸", "轴距",
                "空间", "后备箱", "座椅", "加热", "通风", "按摩", "天窗",
                "音响", "屏幕", "导航", "车机", "互联", "carplay", "语音",
                "灯光", "LED", "氛围灯", "仪表", "抬头显示", "HUD",
                "辅助驾驶", "自动驾驶", "变道辅助", "盲点监测"
            ],

            # 动力/能耗类
            "动力": [
                "动力", "加速", "百公里", "马力", "扭矩", "发动机", "变速箱",
                "油耗", "耗油", "几个油", "省油", "费油", "油费", "加油",
                "续航", "续航里程", "纯电", "混动", "充电", "快充", "慢充",
                "电池", "多少度电", "电耗", "充满", "电能", "油箱", "排量",
                "涡轮", "自吸", "轻混", "插混"
            ],

            # 安全类
            "安全": [
                "安全", "碰撞", "气囊", "气帘", "主动刹车", "自动刹车",
                "防撞", "预警", "车道保持", "车道偏离", "并线辅助",
                "主动安全", "被动安全", "钢", "车身结构", "稳定控制",
                "ESP", "ABS", "360", "全景影像", "雷达", "倒车", "泊车"
            ],

            # 外观/颜色类
            "外观": [
                "颜色", "白色", "黑色", "红色", "蓝色", "灰色", "金色", "银色",
                "紫", "漆", "车漆", "金属漆", "标准漆", "加钱", "选色",
                "外观", "造型", "前脸", "尾部", "轮毂", "轮胎", "车灯",
                "颜值", "好看", "大气", "运动", "时尚", "年轻"
            ],

            # 对比类
            "对比": [
                "和", "对比", "比较", "哪个好", "怎么选", "区别", "差异",
                "vs", "还是", "或", "跟", "比", "强", "好", "优劣"
            ],

            # 推荐类
            "推荐": [
                "推荐", "适合", "哪款", "选哪", "建议", "选什么", "挑哪个",
                "家用", "商务", "上班", "通勤", "接送", "自驾",
                "身高", "体重", "新手", "老司机", "女", "男"
            ],

            # 售后/养车类
            "售后": [
                "保养", "维护", "维修", "保修", "质保", "终身质保", "免费保养",
                "售后服务", "4S店", "网点", "服务", "救援", "道路救援",
                "保险", "上牌", "年检"
            ],

            # 智能/科技类
            "智能": [
                "智能", "科技", "芯片", "8155", "车机", "OTA", "升级",
                "语音控制", "人脸识别", "疲劳检测", "自动泊车",
                "远程控制", "手机APP", "蓝牙", "NFC", "钥匙",
                "大屏", "双屏", "液晶", "数字化"
            ]
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
        for alias in self.car_aliases.keys():
            if alias in clean_lower or self.car_aliases[alias] in clean:
                car = self.car_aliases[alias]
                break
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

        if  not car:
            # 如果识别到了意图（比如推荐、价格、对比），直接用意图做键
            if intent != "通用":
                return f"无车型_{intent}"
            # 如果连意图都没有，就交给大模型去提炼短标签（直接用前5个字兜底）
            return clean[:5]

        return f"{car}_{intent}"

    def _rule_based_summary(self, question):
        """基于规则提炼短句（10字左右，保留具体诉求）"""
        clean = question.strip().rstrip("？?").strip()
        if not clean:
            return "购车咨询"

        # 1. 提取车型（只取前一个匹配到的车型）
        car = None
        clean_lower = clean.lower()
        for alias, full in self.car_aliases.items():
            if alias in clean_lower or full in clean:
                car = full
                break

        # 2. 提取具体诉求关键词（优先级从高到低）
        detail_keywords = [
            # 能耗类
            ("油耗", ["油耗", "几个油", "费油", "省油"]),
            ("续航", ["续航", "充电", "能跑"]),
            # 配置类
            ("音响", ["音响", "丹拿", "BOSE", "喇叭"]),
            ("360全景", ["360", "全景影像", "透明底盘", "倒车影像"]),
            ("座椅", ["座椅加热", "座椅通风", "座椅记忆"]),
            ("天窗", ["天窗", "全景天窗"]),
            ("大屏", ["大屏", "中控屏", "屏幕"]),
            # 安全类
            ("主动刹车", ["主动刹车", "自动刹车", "防撞"]),
            ("辅助驾驶", ["车道保持", "自适应巡航", "ACC", "L2", "智驾"]),
            ("安全", ["安全", "气囊", "车身"]),
            # 外观类
            ("颜色", ["颜色", "漆", "灰色", "紫色", "红色", "白色", "黑色"]),
            ("外观", ["外观", "造型", "前脸"]),
            # 空间类
            ("空间", ["空间", "轴距", "后备箱", "后排"]),
            # 动力类
            ("动力", ["动力", "加速", "百公里", "马力"]),
            # 价格类
            ("价格", ["优惠", "落地", "多少钱", "价格", "报价", "分期", "月供"]),
            # 对比类
            ("对比", ["对比", "哪个好", "怎么选", "还是", "vs"]),
            # 推荐类
            ("推荐", ["推荐", "适合", "哪款"]),
        ]

        # 3. 匹配具体诉求（取第一个匹配到的）
        matched_detail = ""
        for label, keywords in detail_keywords:
            for kw in keywords:
                if kw in clean:
                    matched_detail = label
                    break
            if matched_detail:
                break

        # 4. 如果没匹配到具体诉求，fallback到意图
        if not matched_detail:
            # 检查意图
            for intent_name, keywords in self.intent_keywords.items():
                for kw in keywords:
                    if kw in clean:
                        matched_detail = intent_name
                        break
                if matched_detail:
                    break

        # 5. 构造短句
        if car:
            # 如果有车型，构造 "车型 诉求" 或 "车型问诉求"
            if matched_detail and matched_detail not in ["通用", "配置", "推荐", "对比", "价格"]:
                summary = f"{car} {matched_detail}"
            elif matched_detail:
                summary = f"{car} {matched_detail}"
            else:
                summary = car
        else:
            # 没有车型，用核心关键词
            if matched_detail:
                summary = f"红旗 {matched_detail}"
            else:
                # 截断保留核心
                clean_core = re.sub(r'请问|你好|我想|想要|推荐|适合|怎么样|好不好|麻烦问一下', '', clean)
                if len(clean_core) > 10:
                    summary = clean_core[:10] + "..."
                else:
                    summary = clean_core if clean_core else "购车咨询"

        # 6. 限制长度（10字左右，允许±2字）
        if len(summary) > 14:
            summary = summary[:12] + "..."
        return summary


    def _summarize_with_ai(self, question, client=None):
        """
        双保险提炼：
        第一道：大模型提炼（聪明）。
        第二道：如果大模型异常/没传client，走本地规则兜底（稳定）。
        """
        clean = question.strip().rstrip("？?").strip()
        if not clean:
            return "购车咨询"

        # 第一道防线：大模型提炼
        if client:
            try:
                prompt = f"""你是一个汽车销售系统的标题助手。请将用户的提问提炼成10个字以内的核心短标题，保留“车型”和“核心意图”。
要求：只输出提炼后的短标题，不要任何标点符号和解释。
例如：
用户问：红旗H5的1.5T发动机市区堵车到底几个油？
输出：红旗H5 油耗

用户提问：{clean}
输出："""
                response = client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=20
                )
                summary = response.choices[0].message.content.strip()
                summary = summary.replace("输出：", "").replace("标题：", "").strip()

                if summary and len(summary) <= 20:
                    return summary
            except Exception as e:
                # 大模型挂了，静默记录错误，准备走兜底
                print(f"[提示] AI提炼失败，启用本地规则兜底: {e}")

        # 第二道防线：本地字典兜底（保留了你之前写的 _rule_based_summary）
        return self._rule_based_summary(clean)
    def _get_best_display_text(self, question, normalized_key):
        """从同组问题中选一个展示文本（保留原用法）"""
        candidates = self.data["questions"].get(normalized_key, {}).get("variants", [])
        if not candidates:
            return question
        candidates.sort(key=len)
        mid = len(candidates) // 2
        return candidates[mid]

    def record_question(self, question, client=None):

        clean = question.strip().rstrip("？?").strip()
        if not clean:
            return

        normalized_key = self._normalize_question(clean)

        #  强制生成 display（大模型优先，失败自动走规则）
        display_name = self._summarize_with_ai(clean, client=client)
        if len(display_name) > 20:
            display_name = display_name[:18] + "..."

        if normalized_key not in self.data["questions"]:
            self.data["questions"][normalized_key] = {
                "count": 0,
                "variants": [],
                "latest": clean,
                "display": display_name  #  关键：必须写入
            }
        else:
            # 如果已有 display，检查是否需要更新
            existing = self.data["questions"][normalized_key].get("display")
            if not existing or len(display_name) < len(existing):
                self.data["questions"][normalized_key]["display"] = display_name

        self.data["questions"][normalized_key]["count"] += 1
        if clean not in self.data["questions"][normalized_key]["variants"]:
            self.data["questions"][normalized_key]["variants"].append(clean)
        self.data["questions"][normalized_key]["latest"] = clean

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
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        to_delete = [d for d in self.data["daily"].keys() if d < cutoff]
        for d in to_delete:
            del self.data["daily"][d]

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

    def get_top_questions(self, n=5, client=None):
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

                # ⭐ 强制实时提炼，忽略 JSON 里可能存错的 display
                if variants:
                    display = self._summarize_with_ai(variants[0], client=client)
                else:
                    display = self._summarize_with_ai(key, client=client)
            else:
                count = info
                display = self._summarize_with_ai(key, client=client)

            # 强制截断
            if len(display) > 20:
                display = display[:18] + "..."

            if display in seen:
                continue
            seen.add(display)
            result.append(display)
            if len(result) >= n:
                break
        return result