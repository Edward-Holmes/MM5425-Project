import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_svm_model(data_link):
    """
    训练支持向量机模型并保存到本地
    """
    try:
        # 创建目录
        os.makedirs('./SVM/model', exist_ok=True)

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
        
        # 数据标准化（SVM对特征尺度敏感）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.1, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        # 创建并训练SVM模型
        print("开始训练支持向量机模型...")
        
        # 使用网格搜索寻找最佳参数（可选）
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        # 简单的SVM训练（取消注释下面的代码使用网格搜索）
        svm_model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=False,  # 我们不使用概率估计，使用决策函数
            random_state=42
        )
        
        # 如果需要网格搜索，取消注释以下代码
        """
        svm_model = GridSearchCV(
            SVC(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        """
        
        svm_model.fit(X_train, y_train)
        
        # 如果使用了网格搜索，显示最佳参数
        if hasattr(svm_model, 'best_params_'):
            print(f"最佳参数: {svm_model.best_params_}")
            svm_model = svm_model.best_estimator_
        
        print("SVM模型训练完成!")
        
        # 在测试集上评估模型
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 显示支持向量信息
        print(f"\n支持向量数量: {len(svm_model.support_vectors_)}")
        print(f"支持向量占总样本比例: {len(svm_model.support_vectors_)/len(X_train):.2%}")
        
        # 保存模型到本地
        model_filename = './SVM/model/svm_model.pkl'
        joblib.dump(svm_model, model_filename)
        print(f"\nSVM模型已保存到: {model_filename}")
        
        # 保存标准化器
        scaler_filename = './SVM/model/scaler.pkl'
        joblib.dump(scaler, scaler_filename)
        print(f"标准化器已保存到: {scaler_filename}")
        
        # 保存特征名称
        feature_info = {
            'features': X.columns.tolist(),
            'scaler_info': '数据已标准化，预测时需使用相同标准化器'
        }
        joblib.dump(feature_info, './SVM/model/model_features.pkl')
        print("特征信息已保存")
        
        # 显示模型参数
        print("\n模型参数:")
        print(f"核函数: {svm_model.kernel}")
        print(f"惩罚参数C: {svm_model.C}")
        print(f"gamma参数: {svm_model.gamma}")
        
        return svm_model, scaler, X_test, y_test
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return None, None, None, None

def analyze_decision_function(model, X_sample, feature_names):
    """
    分析决策函数值（到超平面的距离）
    """
    if hasattr(model, 'decision_function'):
        distances = model.decision_function(X_sample)
        print("\n决策函数分析:")
        print(f"样本到超平面的距离: {distances}")
        
        # 对于线性核，可以获取权重向量
        if model.kernel == 'linear':
            print(f"权重向量: {model.coef_}")
            print(f"偏置项: {model.intercept_}")
    
    return distances

if __name__ == "__main__":
    model, scaler, X_test, y_test = train_svm_model('./data.csv')
    
    if model is not None:
        print("\nSVM训练脚本执行完毕，模型已保存！")
        print("可以使用测试脚本进行预测。")
        
        # 示例：分析一个测试样本
        if X_test is not None and len(X_test) > 0:
            sample = X_test[0:1]  # 取第一个测试样本
            print(f"\n测试样本特征值: {sample}")
            distances = analyze_decision_function(model, sample, ['speed', 'taste', 'service'])
    else:
        print("\n训练失败，请检查数据文件格式。")