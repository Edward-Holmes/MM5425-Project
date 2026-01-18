import pandas as pd
import datetime

# 读取CSV文件
df = pd.read_csv('./Raw_Data/天天煲仔饭.csv', encoding='gbk')

# 确保我们有所需的列
print("原始数据列名:", df.columns.tolist())
print(f"原始数据记录数: {len(df)}")

# 第一步：删除空白评论
# 假设评论内容在第5列（索引4），根据实际情况调整
comment_col_index = 4

# 删除评论内容为空的行（包括NaN、空字符串等）
df_cleaned = df.dropna(subset=[df.columns[comment_col_index]])  # 删除NaN
df_cleaned = df_cleaned[df_cleaned.iloc[:, comment_col_index].astype(str).str.strip() != '']  # 删除空字符串

print(f"删除空白评论后记录数: {len(df_cleaned)}")

# 提取所需数据
shop_name = df_cleaned.iloc[0, 0]  # 获取店铺名称（假设在第一列第一行）

# 处理评论时间转换
def timestamp_to_datetime(timestamp):
    """将时间戳转换为标准时间格式"""
    try:
        # 假设时间戳是秒级（如果是毫秒级需要除以1000）
        dt = datetime.datetime.fromtimestamp(int(timestamp))
        return dt.strftime("%Y/%m/%d %H:%M:%S")
    except:
        return timestamp

# 计算评分均分（假设评分在第4列，索引为3）
ratings = df_cleaned.iloc[:, 3].astype(float)  # 评分列
average_score = ratings.mean()
print(f"评分均分: {average_score:.2f}")

# 创建新的DataFrame，包含需要的列
# 假设列索引：评分(3), 评论内容(4), 评论时间(5)
new_df = pd.DataFrame({
    'score': df_cleaned.iloc[:, 3],
    'comment': df_cleaned.iloc[:, 4],
    'timestamp': df_cleaned.iloc[:, 5].apply(timestamp_to_datetime)
})

# 显示前几行数据
print("\n处理后的数据前5行:")
print(new_df.head())

# 保存到新的CSV文件，文件名为店铺名称
output_filename = f"./Process_Data/{shop_name}_processed.csv"
new_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n数据已保存到: {output_filename}")
print(f"总记录数: {len(new_df)}")