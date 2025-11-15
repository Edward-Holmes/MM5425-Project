import pandas as pd
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_gnb_model(data_link):
    """
    训练高斯朴素贝叶斯模型并保存到本地
    """
    try:
        # 创建目录
        os.makedirs('./GNB/model', exist_ok=True)

        # 读取数据文件
        print("正在读取数据文件...")
        df = pd.read_csv(data_link)
        print(f"数据读取成功，共 {len(df)} 条记录")
        
        # 检查数据列
        print("数据列:", df.columns.tolist())
        
        # 提取特征和标签
        X = df[['speed', 'taste', 'service']]
        y = df['label']
        
        print(f"特征数据形状: {X.shape}")
        print(f"标签分布:\n{y.value_counts()}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        # 创建并训练高斯朴素贝叶斯模型
        print("开始训练高斯朴素贝叶斯模型...")
        gnb_model = GaussianNB()
        
        gnb_model.fit(X_train, y_train)
        print("模型训练完成!")
        
        # 在测试集上评估模型
        y_pred = gnb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 保存模型到本地
        model_filename = './GNB/model/gaussian_nb_model.pkl'
        joblib.dump(gnb_model, model_filename)
        print(f"\n模型已保存到: {model_filename}")
        
        # 保存特征信息（用于预测时验证）
        feature_info = {
            'features': X.columns.tolist(),
            'class_labels': gnb_model.classes_.tolist()
        }
        joblib.dump(feature_info, './GNB/model/model_features.pkl')
        print("特征信息已保存")
        
        # 高斯朴素贝叶斯没有特征重要性，但可以查看每个特征的均值和方差
        print("\n各类别的先验概率:")
        for i, class_label in enumerate(gnb_model.classes_):
            print(f"类别 {class_label}: {gnb_model.class_prior_[i]:.4f}")
            
        return gnb_model, X_test, y_test
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    model, X_test, y_test = train_gnb_model('./data.csv')
    
    if model is not None:
        print("\n训练脚本执行完毕，模型已保存！")
        print("可以使用测试脚本进行预测。")
    else:
        print("\n训练失败，请检查数据文件格式。")