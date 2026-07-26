import os
import pandas as pd

log_file = "data/user_logs.csv"
backup_file = "data/user_logs_backup.csv"

if os.path.exists(log_file):
    # 备份原文件
    os.rename(log_file, backup_file)

    # 读取旧数据（无表头）
    df = pd.read_csv(backup_file, encoding='utf-8', header=None)

    # 根据旧格式推断列数
    if len(df.columns) == 6:
        # 旧格式: timestamp, session_id, user_input, ai_response, retrieved_models, matched_cars
        df.columns = ['timestamp', 'session_id', 'user_input', 'ai_response', 'retrieved_models', 'matched_cars']
        # 新增 customer_id 列（默认空）
        df['customer_id'] = ""
        # 重新排列
        df = df[
            ['timestamp', 'session_id', 'customer_id', 'user_input', 'ai_response', 'retrieved_models', 'matched_cars']]
    else:
        # 如果有其他列，手动调整
        print("⚠️ 列数不匹配，请检查日志格式")

    # 保存带表头的新文件
    df.to_csv(log_file, encoding='utf-8', index=False)
    print(f"✅ 迁移完成！原文件已备份为 {backup_file}")
else:
    print("❌ 日志文件不存在，无需迁移")