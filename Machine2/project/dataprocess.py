import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from xgboost.testing.data import joblib

def remove_id_column(data, id_column='ID'):
    """
    去除指定的 ID 列。
    """
    if id_column in data.columns:
        data = data.drop(columns=[id_column])
    return data

def remove_pcr_outcome_999(data):
    """
    剔除 pCR (outcome) 列中值为 999 的整条数据，因为这列是目标值缺失，不进行数据填充与处理。
    """
    return data[data['pCR (outcome)'] != 999]

def knn_impute_missing_values(data, n_neighbors=5):
    """
    使用 KNN 方法填补缺失值（999 被视为缺失值）。
    """
    # 替换 999 为 NaN
    data = data.replace(999, np.nan)

    # 使用 KNNImputer 填补缺失值
    imputer = KNNImputer(n_neighbors=n_neighbors)
    data_imputed = imputer.fit_transform(data)

    # 转回 DataFrame，保持原始列名
    data_filled = pd.DataFrame(data_imputed, columns=data.columns)
    return data_filled

def separate_features_and_targets(data, classification_target_column, regression_target_column):
    """
    将数据集分为特征和两个标签（一个用于分类，一个用于回归）。
    """
    X = data.drop(columns=[classification_target_column, regression_target_column])
    y_classification = data[classification_target_column]
    y_regression = data[regression_target_column]
    return X, y_classification, y_regression

def standardize_features(X):
    """
    标准化特征。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess_data(data, classification_target_column, regression_target_column, n_neighbors=5):
    """
    预处理数据，包括剔除特定值、使用 KNN 填补缺失值、分离特征和目标变量及标准化。
    """
    data = remove_id_column(data)  # 去除 ID 列
    data = remove_pcr_outcome_999(data)  # 删除 pCR (outcome) 中值为 999 的行
    data = knn_impute_missing_values(data, n_neighbors=n_neighbors)  # 使用 KNN 填补缺失值

    X, y_classification, y_regression = separate_features_and_targets(
        data, classification_target_column, regression_target_column
    )
    output_file_path = 'data/processed/ProcessedDataset2024.csv'
    data.to_csv(output_file_path, index=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 结果是 NumPy 数组

    # 转回 Pandas DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # 模型保存训练时的 scaler
    joblib.dump(scaler, "scaler.pkl")
    return X_scaled, y_classification, y_regression
