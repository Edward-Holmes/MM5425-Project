import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
import time
import warnings
warnings.filterwarnings('ignore')

class SVMTester:
    def __init__(self, model_path='./SVM/model/svm_model.pkl', feature_path='./SVM/model/model_features.pkl'):
        """
        初始化SVM测试器，加载模型和特征信息
        """
        try:
            self.model = joblib.load(model_path)
            self.feature_info = joblib.load(feature_path)
            self.features = self.feature_info['features']
            print(f"SVM模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            self.model = None
    
    def get_hyperplane_info(self, X):
        """
        获取超平面相关信息
        Args:
            X: 输入特征数据
        Returns:
            dict: 包含决策函数值、支持向量信息等
        """
        if self.model is None:
            print("模型未加载成功，无法获取超平面信息")
            return None
        
        # 获取决策函数值（到超平面的距离）
        decision_values = self.model.decision_function(X)
        
        # 获取支持向量信息
        support_vectors = self.model.support_vectors_
        support_vector_indices = self.model.support_
        
        hyperplane_info = {
            'decision_values': decision_values,
            'support_vectors': support_vectors,
            'support_vector_indices': support_vector_indices,
            'n_support_vectors': len(support_vectors),
            'distance_to_hyperplane': decision_values
        }
        
        return hyperplane_info
    
    def predict_single_cases(self, test_cases):
        """
        对单个测试用例进行预测，返回超平面信息
        Args:
            test_cases: list of tuples, 每个tuple包含(speed, taste, service)
        Returns:
            predictions: 预测结果列表
            hyperplane_info: 超平面信息字典
        """
        if self.model is None:
            print("模型未加载成功，无法进行预测")
            return None, None
        
        # 转换为DataFrame
        test_df = pd.DataFrame(test_cases, columns=self.features)
        
        # 进行预测
        predictions = self.model.predict(test_df)
        
        # 获取超平面信息
        hyperplane_info = self.get_hyperplane_info(test_df)
        
        print("\n" + "="*60)
        print("SVM单样本预测结果")
        print("*"*60)
        
        for i, (test_case, pred, dist) in enumerate(zip(test_cases, predictions, hyperplane_info['distance_to_hyperplane'])):
            print(f"\n测试用例 {i+1}: {test_case}")
            print(f"预测结果: {pred}")
            # print(f"到超平面的距离: {dist:.4f}")
            
            # 对于多分类问题，显示每个类别的距离
            if len(dist) > 1:
                for j, class_dist in enumerate(dist):
                    print(f"  到类别 {self.model.classes_[j]} 的超平面距离: {class_dist:.4f}")
        
        return predictions, hyperplane_info
    
    def batch_predict(self, test_df, label_column='label'):
        """
        批量预测并评估模型性能
        Args:
            test_df: DataFrame, 测试数据
            label_column: str, 标签列名
        Returns:
            dict: 包含各项评估指标和预测结果
        """
        if self.model is None:
            print("模型未加载成功，无法进行预测")
            return None
        
        # 检查特征列是否存在
        missing_features = set(self.features) - set(test_df.columns)
        if missing_features:
            print(f"数据框中缺少特征列: {missing_features}")
            return None
        
        # 提取特征和真实标签
        X_test = test_df[self.features]
        if label_column in test_df.columns:
            y_true = test_df[label_column]
            has_true_labels = True
        else:
            y_true = None
            has_true_labels = False
        
        # 记录开始时间
        start_time = time.time()
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        
        # 获取超平面信息
        hyperplane_info = self.get_hyperplane_info(X_test)
        
        # 计算用时
        time_used = time.time() - start_time
        
        # 准备结果
        result = {
            'predictions': y_pred,
            'hyperplane_info': hyperplane_info,
            'time_used': time_used,
            'class_labels': self.model.classes_,
            'n_samples': len(X_test)
        }
        
        # 如果有真实标签，计算评估指标
        if has_true_labels:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            result.update({
                'true_labels': y_true,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        return result

# 编程接口函数
def load_SVM(model_path='./SVM/model/svm_model.pkl', feature_path='./SVM/model/model_features.pkl'):
    """
    加载训练好的SVM模型
    Returns:
        SVMTester实例
    """
    return SVMTester(model_path, feature_path)

def predict_single(model_tester, test_cases):
    """
    对单个测试用例进行预测的接口
    Args:
        model_tester: SVMTester实例
        test_cases: 测试用例列表
    """
    return model_tester.predict_single_cases(test_cases)

def predict_batch(model_tester, test_data, label_column='label'):
    """
    批量预测的接口
    Args:
        model_tester: SVMTester实例
        test_data: 测试数据DataFrame或文件路径
        label_column: 标签列名
    """
    if isinstance(test_data, str):
        # 如果是文件路径，加载数据
        test_df = pd.read_csv(test_data)
    else:
        test_df = test_data
    
    return model_tester.batch_predict(test_df, label_column)

def SVM_Test(test_df):
    """
    SVM批量测试函数
    """
    # 加载模型
    print("="*60)
    print("加载SVM模型中...")
    print("*"*60)
    svm_model = load_SVM()

    if svm_model.model is not None:
        try:
            # 尝试加载测试数据
            batch_result = predict_batch(svm_model, test_df)
            
            return batch_result
                
        except Exception as e:
            print(f"批量预测过程中出现错误: {str(e)}")
            return None
    
    return None

def SVM_Prediction(test_cases):
    """
    SVM单样本预测函数
    """
    # 加载模型
    print("加载SVM模型中...")
    svm_model = load_SVM()

    if svm_model.model is not None:
        predictions, hyperplane_info = predict_single(svm_model, test_cases)
        return predictions, hyperplane_info
    
    return None, None


# 测试示例
if __name__ == "__main__":
    # 批量预测测试
    test_cases1 = pd.read_csv('data.csv')
    batch_result = SVM_Test(test_cases1)
    
    if batch_result is not None and 'true_labels' in batch_result:
        print("\n" + "="*60)
        print("SVM批量预测评估结果")
        print("*"*60)
        print(f"样本数量: {batch_result['n_samples']}")
        print(f"支持向量数量: {batch_result['hyperplane_info']['n_support_vectors']}")
        print(f"Time Used: {batch_result['time_used']:.4f} 秒")
        print(f"Accuracy: {batch_result['accuracy']:.4f}")
        print(f"Precision Rate: {batch_result['precision']:.4f}")
        print(f"Recall: {batch_result['recall']:.4f}")
        print(f"F1 Score: {batch_result['f1_score']:.4f}")
        
        # 显示前几个样本的超平面距离
        print("\n前5个样本的超平面信息:")
        for i in range(min(5, len(batch_result['predictions']))):
            print(f"样本 {i+1}: 预测={batch_result['predictions'][i]}, " 
                  f"到超平面距离={batch_result['hyperplane_info']['distance_to_hyperplane'][i]}")
    
    # 单样本预测测试
    test_cases2 = [
        (0.8, 0.7, 0.9),
        (-0.5, -0.5, -0.6),
        (0.1, 0.2, -0.1)
    ]
    
    SVM_Prediction(test_cases2)
    
    print("\nSVM测试脚本执行完毕!")