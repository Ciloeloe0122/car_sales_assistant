import sqlite3

conn = sqlite3.connect("crm.db")
cursor = conn.cursor()

print("=" * 40)
print("【所有客户】")
cursor.execute("SELECT id, phone, name, registered_at FROM customer")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, 手机: {row[1]}, 姓名: {row[2]}, 注册时间: {row[3]}")

print("\n" + "=" * 40)
print("【所有销售顾问】")
cursor.execute("SELECT id, phone, name, created_at FROM sales_rep")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, 手机: {row[1]}, 姓名: {row[2]}, 创建时间: {row[3]}")

conn.close()
print("\n" + "=" * 40)
print("提示：密码是加密存储的，无法直接查看明文。所有测试账号密码均为 123456。")