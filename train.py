import GNB.GNB_Train as GNBT
import RF.RF_Train as RFT
import SVM.SVM_Train as SVMT
import time
import warnings
warnings.filterwarnings('ignore')

def Train_GNB(data_link):
    """训练高斯朴素贝叶斯模型"""
    print("="*60)
    print("开始训练高斯朴素贝叶斯模型...")
    print("*"*60)
    
    # 使用正确的函数名 train_gnb_model
    model, X_test, y_test = GNBT.train_gnb_model(data_link)
        
    if model is not None:
        print("\n✓ 高斯朴素贝叶斯模型训练完成并已保存！")
    else:
        print("\n✗ 高斯朴素贝叶斯模型训练失败")
    return model

def Train_RF(data_link):
    """训练随机森林模型"""
    print("="*60)
    print("开始训练随机森林模型...")
    print("*"*60)
    
    # 使用正确的函数名 train_rf_model
    model, X_test, y_test = RFT.train_rf_model(data_link)
        
    if model is not None:
        print("\n✓ 随机森林模型训练完成并已保存！")
    else:
        print("\n✗ 随机森林模型训练失败")
    return model

def Train_SVM(data_link):
    """训练支持向量机模型"""
    print("="*60)
    print("开始训练支持向量机模型...")
    print("*"*60)
    
    # 使用正确的函数名 train_svm_model
    # 注意：SVM返回4个值，其他返回3个值
    model, scaler, X_test, y_test = SVMT.train_svm_model(data_link)
        
    if model is not None:
        print("\n✓ 支持向量机模型训练完成并已保存！")
    else:
        print("\n✗ 支持向量机模型训练失败")
    return model

def Train_All(data_link):
    """训练所有模型"""
    print("="*60)
    print("开始训练所有机器学习模型")
    print("*"*60)
    
    start_time = time.time()
    
    # 训练三个模型
    models = {}
    models['GNB'] = Train_GNB(data_link)
    models['RF'] = Train_RF(data_link)
    models['SVM'] = Train_SVM(data_link)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("所有模型训练完成！")
    print(f"总训练时间: {total_time:.2f} 秒")
    print("*"*60)
    
    # 检查训练结果
    success_count = sum(1 for model in models.values() if model is not None)
    print(f"成功训练模型数量: {success_count}/3")
    
    return models

def Train_Single(model_name, data_link):
    """训练单个指定模型"""
    model_name = model_name.upper()
    
    if model_name == 'GNB':
        return Train_GNB(data_link)
    elif model_name == 'RF':
        return Train_RF(data_link)
    elif model_name == 'SVM':
        return Train_SVM(data_link)
    else:
        print(f"未知模型名称: {model_name}")
        print("支持的模型: GNB, RF, SVM")
        return None

if __name__ == "__main__":
    data_link = './T_Data/train_data.csv'
    
    # 使用方法1: 训练所有模型
    Train_All(data_link)
    
    # 使用方法2: 训练单个模型
    # Train_Single('RF', data_link)