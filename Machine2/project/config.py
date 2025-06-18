# 配置文件

# 数据文件路径
DATA_FILE_PATH = './data/original/TrainDataset2024.xls'
PROCESSED_FILE_PATH = "data/processed/ProcessedDataset2024.csv"  # 处理后的数据保存路径

# 分类和回归目标列名
CLASSIFICATION_TARGET_COLUMN = 'pCR (outcome)'  # 实际的分类目标列名
REGRESSION_TARGET_COLUMN = 'RelapseFreeSurvival (outcome)'  # 实际的回归目标列名

# 模型保存路径
RANDOM_FOREST_MODEL_PATH = "models/random_forest.pkl"
GRADIENT_BOOSTING_MODEL_PATH = "models/gradient_boosting.pkl"
LINEAR_REGRESSION_MODEL_PATH = "models/linear_regression.pkl"
SVM_MODEL_PATH = "models/svm.pkl"
XGBOOST_MODEL_PATH="models/xgboost_model.pkl"
LIGHTGBM_MODEL_PATH="models/lightgbm_model.pkl"
MLP_MODEL_PATH="models/mlp_model.pkl"

#测试结果保存路径
MODEL_OUTPUT_PATH="results/model_outputs"

#模型表现保存路径
PERFORMANCE_REPORT_PATH="results/performance_reports"

# 随机种子
RANDOM_STATE = 42

# 数据分割比例
TEST_SIZE = 0.2  # 测试集占比

# 超参数设置
RANDOM_FOREST_PARAMS = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 10],
}

SVM_PARAMS = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "epsilon": [0.1, 0.2, 0.5],
}

