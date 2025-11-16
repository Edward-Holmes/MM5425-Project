import GNB.GNB_Test as GNB_T
import RF.RF_Test as RF_T
import SVM.SVM_Test as SVM_T
import pandas as pd
import Vis.vis as vis

def Test_CSV_GNB(data):
    # 批量预测测试
    test_cases1 = pd.read_csv(data)
    batch_result = GNB_T.GNB_Test(test_cases1)
    
    if batch_result is not None and 'true_labels' in batch_result:
        
        print("\n" + "="*60)
        print("GNB批量预测评估结果")
        print("*"*60)
        print(f"样本数量: {batch_result['n_samples']}")
        print(f"Time Used: {batch_result['time_used']:.4f} 秒")
        print(f"Accuracy: {batch_result['accuracy']:.4f}")
        print(f"Precision Rate: {batch_result['precision']:.4f}")
        print(f"Recall: {batch_result['recall']:.4f}")
        print(f"F1 Score: {batch_result['f1_score']:.4f}")
        
        # 详细分类报告
        print("\n详细分类报告:")
        print(GNB_T.classification_report(batch_result['true_labels'], batch_result['predictions'], zero_division=0))
        
        print(f"\n批量预测完成!")
        print(f"处理了 {batch_result['n_samples']} 个样本")
        print(f"平均每个样本处理时间: {batch_result['time_used']/batch_result['n_samples']*1000:.2f} 毫秒")
        return batch_result

def Test_CSV_RF(data):
    # 批量预测测试
    test_cases1 = pd.read_csv(data)
    batch_result = RF_T.RF_Test(test_cases1)
    
    if batch_result is not None and 'true_labels' in batch_result:
        print("\n" + "="*60)
        print("RF批量预测评估结果")
        print("*"*60)
        print(f"样本数量: {batch_result['n_samples']}")
        print(f"Time Used: {batch_result['time_used']:.4f} 秒")
        print(f"Accuracy: {batch_result['accuracy']:.4f}")
        print(f"Precision Rate: {batch_result['precision']:.4f}")
        print(f"Recall: {batch_result['recall']:.4f}")
        print(f"F1 Score: {batch_result['f1_score']:.4f}")
        return batch_result

def Test_CSV_SVM(data):
    # 批量预测测试
    test_cases1 = pd.read_csv(data)
    batch_result = SVM_T.SVM_Test(test_cases1)
    
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
        
        return batch_result

def Test_All(test_cases2):
    GNB_T.GNB_Prediction(test_cases2)
    RF_T.RF_Prediction(test_cases2)
    SVM_T.SVM_Prediction(test_cases2)

def Test_CSV_All(data):
    GNB_batch_result = Test_CSV_GNB(data)
    RF_batch_result = Test_CSV_RF(data)
    SVM_batch_result = Test_CSV_SVM(data)

    accuracy = [GNB_batch_result['accuracy'], RF_batch_result['accuracy'], SVM_batch_result['accuracy']]
    time_used = [GNB_batch_result['time_used'], RF_batch_result['time_used'], SVM_batch_result['time_used']]
    precision_rate = [GNB_batch_result['precision'], RF_batch_result['precision'], SVM_batch_result['precision']]
    recall = [GNB_batch_result['recall'], RF_batch_result['recall'], SVM_batch_result['recall']]
    F1 = [GNB_batch_result['f1_score'], RF_batch_result['f1_score'], SVM_batch_result['f1_score']]
    n_samples = [GNB_batch_result['n_samples'], RF_batch_result['n_samples'], SVM_batch_result['n_samples']]

    return accuracy, time_used, precision_rate, recall, F1, n_samples

if __name__ == "__main__":
    # 单样本预测测试
    test_cases2 = [
        (0.8, 0.7, 0.9),
        (-0.5, -0.5, -0.6),
        (0.1, 0.2, -0.1)
    ]

    data = './T_Data/test_data.csv'
    accuracy, time_used, precision_rate, recall, F1, n_samples = Test_CSV_All(data)

    plt_data = [
        accuracy,
        precision_rate,
        recall,
        F1,
        time_used
    ]

    model_groups = ['Accuracy', 'Precision', 'Recall', 'F1', 'Time Used']
    model_labels = ['GNB', 'RF', 'SVM']

    vis.Vis_Data(data=plt_data, groups=model_groups, bar_labels=model_labels, 
             title='Custom Model Comparison - Metrics')

    Test_All(test_cases2)