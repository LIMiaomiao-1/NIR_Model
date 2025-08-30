import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
from data_processor import DataProcessor  # 确保 DataProcessor 类的路径正确

def load_data(file_path, is_train=True):
    df = pd.read_excel(file_path, header=1)  # 从第二行开始读取数据
    if is_train:
        X = df.iloc[:, 0:2074].values  # 波长数，1-2074列
        y = df.iloc[:, 2074].values.ravel()  # 化学成分，第2075列
        sample_ids = df.iloc[:, 2075].values  # 样品序号，第2076列
    else:
        X = df.iloc[:, 0:2074].values  # 波长数，1-2074列
        sample_ids = df.iloc[:, 2074].values  # 样品序号，第2075列
        y = None
    return X, y, sample_ids

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    sep = np.sqrt(np.sum((y_true - y_pred) ** 2) / (len(y_true) - 1))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, r2, sep, mae

if __name__ == "__main__":
    # 加载数据
    X_train, y_train, sample_ids_train = load_data(r"C:\Users\Desktop\近红外建模\近红外光谱数据与说明-2025年\校正集-总.xlsx", is_train=True)
    X_test, y_test, sample_ids_test = load_data(r"C:\Users\Desktop\近红外建模\近红外光谱数据与说明-2025年\测试集-总.xlsx", is_train=False)

    # 初始化数据处理器
    processor = DataProcessor(X_train, y_train)

    # 定义预处理和特征选择方法
    preprocess_methods = ['derivative']  # 可以添加其他预处理方法
    feature_selection_methods = ['recursive_elimination']  

    # 定义采样数目
    count = 400  

    # 处理数据
    processed_X_train = X_train.copy()
    processed_X_test = X_test.copy()

    if 'airPLS' in preprocess_methods:
        print("应用airPLS基线校正...")
        processed_X_train = processor.baseline_correction_airPLS(processed_X_train, lambda_=100, itermax=15)
        processed_X_test = processor.baseline_correction_airPLS(processed_X_test, lambda_=100, itermax=15)

    if 'MSC' in preprocess_methods:
        print("应用多元散射校正...")
        processed_X_train = processor.perform_msc(processed_X_train)
        processed_X_test = processor.perform_msc(processed_X_test)

    if 'SNV' in preprocess_methods:
        print("应用标准正态变量变换...")
        processed_X_train = processor.perform_standard_normal_variate(processed_X_train)
        processed_X_test = processor.perform_standard_normal_variate(processed_X_test)

    if 'Savitzky-Golay' in preprocess_methods:
        print("应用Savitzky-Golay平滑滤波...")
        processed_X_train = processor.perform_savgol(processed_X_train)
        processed_X_test = processor.perform_savgol(processed_X_test)

    if 'detrend' in preprocess_methods:
        print("应用去趋势...")
        processed_X_train = processor.perform_detrend(processed_X_train)
        processed_X_test = processor.perform_detrend(processed_X_test)

    if 'derivative' in preprocess_methods:
        print("一阶微分")
        processed_X_train = processor.spectral_first_order_derivative(processed_X_train)
        processed_X_test = processor.spectral_first_order_derivative(processed_X_test)

    # 应用特征选择方法
    for method in feature_selection_methods:
        if method == 'uniform_sampling':
            print(f"应用均匀采样 (count={count})...")
            sampled_train, sampled_test = processor.uniform_sampling(
                processed_X_train, 
                processed_X_test, 
                count=count  
            )
            processed_X_train = sampled_train
            processed_X_test = sampled_test
        elif method == 'univariant_selection':
            print("应用单变量特征选择...")
            selected_indices = processor.perform_univariant_selection(processed_X_train, processor.y, count=count)
            processed_X_train = processed_X_train[:, selected_indices]
            processed_X_test = processed_X_test[:, selected_indices]
        elif method == 'recursive_elimination':
            print("应用递归特征消除...")
            selected_indices = processor.perform_recursive_elimination(processed_X_train, processor.y, count=count)
            processed_X_train = processed_X_train[:, selected_indices]
            processed_X_test = processed_X_test[:, selected_indices]
        elif method == 'pca':
            print("应用主成分分析...")
            pca_train = processor.perform_pca(processed_X_train, count=count)
            pca_test = processor.perform_pca(processed_X_test, count=count)
            processed_X_train = pca_train
            processed_X_test = pca_test
        elif method == 'uve':
            print("应用无信息变量消除...")
            selected_indices = processor.perform_uve(processed_X_train, processor.y, count=count)
            processed_X_train = processed_X_train[:, selected_indices]
            processed_X_test = processed_X_test[:, selected_indices]

    print("数据处理完成！")

    # 定义模型
    models = {
        'PLSR': PLSRegression(),
        'TabPFN': TabPFNRegressor(
            device='cpu',
            ignore_pretraining_limits=True,
            inference_precision=torch.float32,
            memory_saving_mode=True,
            fit_mode='fit_preprocessors',
            random_state=42
        )
    }

    # 训练和评估模型
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        if model_name == 'PLSR':
            # 使用网格搜索调整主成分数
            param_grid = {'n_components': [5, 10, 15, 20, 25, 30, 35, 40]}
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
            grid_search.fit(processed_X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
        else:
            best_model = model
            best_params = 'Default params'
            best_model.fit(processed_X_train, y_train)
        
        if y_train is not None:
            y_pred_train = best_model.predict(processed_X_train)
            rmse_train, r2_train, sep_train, mae_train = evaluate_model(y_train, y_pred_train)
            print(f"{model_name} - RMSE (Train): {rmse_train:.4f}, R² (Train): {r2_train:.4f}, SEP (Train): {sep_train:.4f}, MAE (Train): {mae_train:.4f}")

        y_pred_test = best_model.predict(processed_X_test)
        results[model_name] = {'Predicted': y_pred_test}
        print(f"{model_name} - Predictions on test set: {y_pred_test[:5]}")  # 打印前5个预测值

    # 保存预测结果到Excel
    for model_name, model_results in results.items():
        predictions_df = pd.DataFrame({
            'Sample_ID': sample_ids_test,
            'Predicted': model_results['Predicted']
        })
        predictions_df.to_excel(f'predictions_{model_name}.xlsx', index=False)


    print("Predictions saved to 'predictions_{model_name}.xlsx' files.")
