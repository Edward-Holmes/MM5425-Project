import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_rf_model(data_link):
    """
    训练随机森林模型并保存到本地
    """
    try:
        # 创建目录
        os.makedirs('./RF/model', exist_ok=True)

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
        
        # 创建并训练随机森林模型
        print("开始训练随机森林模型...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        print("模型训练完成!")
        
        # 在测试集上评估模型
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 保存模型到本地
        model_filename = './RF/model/random_forest_model.pkl'
        joblib.dump(rf_model, model_filename)
        print(f"\n模型已保存到: {model_filename}")
        
        # 保存特征名称（用于预测时验证）
        feature_info = {
            'features': X.columns.tolist(),
            'feature_importance': dict(zip(X.columns, rf_model.feature_importances_))
        }
        joblib.dump(feature_info, './RF/model/model_features.pkl')
        print("特征信息已保存")
        
        print("\n特征重要性:")
        for feature, importance in sorted(zip(X.columns, rf_model.feature_importances_), 
                                        key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
            
        return rf_model, X_test, y_test
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    model, X_test, y_test = train_rf_model('./data.csv')
    
    if model is not None:
        print("\n训练脚本执行完毕，模型已保存！")
        print("可以使用第二个脚本进行预测。")
    else:
        print("\n训练失败，请检查数据文件格式。")