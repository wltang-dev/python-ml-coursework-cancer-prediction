from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

from Machine2.project.config import DATA_FILE_PATH, CLASSIFICATION_TARGET_COLUMN, REGRESSION_TARGET_COLUMN, \
    RANDOM_STATE, GRADIENT_BOOSTING_PARAMS, MODEL_OUTPUT_PATH, RANDOM_FOREST_PARAMS, PERFORMANCE_REPORT_PATH, \
    RANDOM_FOREST_MODEL_PATH, GRADIENT_BOOSTING_MODEL_PATH, LINEAR_REGRESSION_MODEL_PATH, SVM_MODEL_PATH, \
    XGBOOST_MODEL_PATH, LIGHTGBM_MODEL_PATH, MLP_MODEL_PATH
from Machine2.project.dataprocess import preprocess_data


def evaluate_model(model_name, y_test, y_pred):
    """
    评估模型性能
    """
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")
    return mse, r2

def save_best_model_predictions(best_model, X_test, y_test, model_name):
    """
    保存最佳模型的预测结果到 CSV 文件
    """
    predictions = best_model.predict(X_test)
    output_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })
    output_path = os.path.join(MODEL_OUTPUT_PATH, f"{model_name}_predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def save_performance_report(results_df):
    """
    保存模型性能报告到文件
    """
    report_path = os.path.join(PERFORMANCE_REPORT_PATH, "model_performance_report.csv")
    results_df.to_csv(report_path, index=False)
    print(f"Performance report saved to {report_path}")

def generate_predictions_for_external_dataset(best_model, dataset_path, output_folder):
    """
    使用最佳模型生成外部测试集的预测结果并保存到文件
    """
    external_data = pd.read_excel(dataset_path)
    # 删除 ID 列
    if 'ID' in external_data.columns:
        external_data = external_data.drop(columns=['ID'])

    # 将缺失值 999 替换为 NaN
    external_data = external_data.replace(999, np.nan)

    # 使用 KNN 填充缺失值
    imputer = KNNImputer(n_neighbors=5)
    external_data = pd.DataFrame(imputer.fit_transform(external_data), columns=external_data.columns)

    # 加载 scaler
    scaler = joblib.load("scaler.pkl")
    X_external = scaler.transform(external_data)
    X_external = pd.DataFrame(X_external, columns=external_data.columns)  # 恢复特征名称

    predictions = best_model.predict(X_external)
    output_df = pd.DataFrame({
        'Predicted': predictions
    })
    output_path = os.path.join(output_folder, "external_test_predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"External test predictions saved to {output_path}")

# 随机森林回归

def train_random_forest(X_train, y_train, kfold):
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        RANDOM_FOREST_PARAMS,
        cv=kfold,
        scoring='r2',
    )
    grid_search.fit(X_train, y_train)
    print("Random Forest Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# 梯度提升回归
def train_gradient_boosting(X_train, y_train, kfold):
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        GRADIENT_BOOSTING_PARAMS,
        cv=kfold,
        scoring='r2',
    )
    grid_search.fit(X_train, y_train)
    print("Gradient Boosting Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# 线性回归
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression does not require parameter tuning.")
    return model

# 支持向量机
def train_svm(X_train, y_train, kfold):
    SVM_PARAMS = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=SVM_PARAMS,
        cv=kfold,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("SVM Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# XGBoost 回归
def train_xgboost(X_train, y_train, kfold):
    XGBOOST_PARAMS = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=XGBRegressor(random_state=RANDOM_STATE),
        param_grid=XGBOOST_PARAMS,
        cv=kfold,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("XGBoost Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# LightGBM 回归
def train_lightgbm(X_train, y_train, kfold):
    LIGHTGBM_PARAMS = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [20, 31, 40],
        'max_depth': [-1, 5, 10]
    }
    grid_search = GridSearchCV(
        estimator=LGBMRegressor(random_state=RANDOM_STATE),
        param_grid=LIGHTGBM_PARAMS,
        cv=kfold,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("LightGBM Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import numpy as np

def train_lasso(X_train, y_train, kfold, alpha=1.0):
    """
    训练 Lasso 回归模型并返回经过交叉验证的模型
    Args:
        X_train (ndarray): 训练数据的特征
        y_train (ndarray): 训练数据的目标
        kfold (KFold): 交叉验证分割器
        alpha (float): Lasso 正则化强度（默认值为 1.0）
    Returns:
        model (Lasso): 训练好的 Lasso 回归模型
    """
    lasso = Lasso(alpha=alpha)
    cv_scores = cross_val_score(lasso, X_train, y_train, cv=kfold, scoring='r2')
    print(f"Lasso Regression - Cross-Validated R² Scores: {cv_scores}")
    print(f"Lasso Regression - Mean CV R²: {np.mean(cv_scores):.4f}")
    lasso.fit(X_train, y_train)
    return lasso


# 简单神经网络（MLP）
def train_mlp(X_train, y_train, kfold):
    MLP_PARAMS = {
        'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500]
    }
    grid_search = GridSearchCV(
        estimator=MLPRegressor(random_state=RANDOM_STATE),
        param_grid=MLP_PARAMS,
        cv=kfold,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("MLP Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def blended_prediction(models, X_test, weights):
    """
    对多个模型进行加权混合
    Args:
        models (list): 模型列表
        X_test (ndarray): 测试集特征
        weights (list): 每个模型的权重
    Returns:
        ndarray: 加权后的预测值
    """
    blended_preds = np.zeros(X_test.shape[0])
    for model, weight in zip(models, weights):
        blended_preds += model.predict(X_test) * weight
    return blended_preds


if __name__ == "__main__":
    # Step 1: 加载数据
    data = pd.read_excel(DATA_FILE_PATH)
    X_scaled, y_classification, y_regression = preprocess_data(
        data, CLASSIFICATION_TARGET_COLUMN, REGRESSION_TARGET_COLUMN
    )

    # Step 2: 划分训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y_regression, test_size=0.2, random_state=RANDOM_STATE
    )

    # Step 3: 设置交叉验证 (KFold)
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Step 4: 模型训练
    rf_model = train_random_forest(X_train_full, y_train_full, kfold)
    xgb_model = train_xgboost(X_train_full, y_train_full, kfold)
    lgbm_model = train_lightgbm(X_train_full, y_train_full, kfold)
    gb_model = train_gradient_boosting(X_train_full, y_train_full, kfold)
    svm_model = train_svm(X_train_full, y_train_full, kfold)
    mlp_model = train_mlp(X_train_full, y_train_full, kfold)
    lr_model = train_linear_regression(X_train_full, y_train_full)
    lasso_model = train_lasso(X_train_full, y_train_full, kfold, alpha=1.0)

    # Step 5: 测试集评估
    results = []

    for model_name, model in [
        ("Random Forest", rf_model),
        ("XGBoost", xgb_model),
        ("LightGBM", lgbm_model),
        ("Gradient Boosting", gb_model),
        ("SVM", svm_model),
        ("MLP", mlp_model),
        ("Linear Regression", lr_model),
        ("Lasso Regression", lasso_model),
    ]:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": model_name, "MSE": mse, "R²": r2})
        print(f"{model_name} - Test MSE: {mse:.4f}, R²: {r2:.4f}")

    # Step 6: 混合模型
    # 加权平均混合模型
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # 权重示例
    models = [rf_model, xgb_model, lgbm_model, gb_model, mlp_model]  # 选择模型
    blended_preds = blended_prediction(models, X_test, weights)

    # 混合模型评估
    blended_mse = mean_squared_error(y_test, blended_preds)
    blended_r2 = r2_score(y_test, blended_preds)
    print(f"Blended Model - Test MSE: {blended_mse:.4f}, R²: {blended_r2:.4f}")

    # 添加混合模型结果到结果列表
    results.append({"Model": "Blended Model", "MSE": blended_mse, "R²": blended_r2})

    # Step 6: 比较模型性能
    results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
    print("\nModel Comparison:")
    print(results_df)

    # Step 7: 保存所有模型，包括混合模型
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, RANDOM_FOREST_MODEL_PATH)
    joblib.dump(xgb_model, XGBOOST_MODEL_PATH)
    joblib.dump(lgbm_model, LIGHTGBM_MODEL_PATH)
    joblib.dump(gb_model, GRADIENT_BOOSTING_MODEL_PATH)
    joblib.dump(svm_model, SVM_MODEL_PATH)
    joblib.dump(mlp_model, MLP_MODEL_PATH)
    joblib.dump(lr_model, LINEAR_REGRESSION_MODEL_PATH)
    joblib.dump(lasso_model, "models/lasso_model.pkl")
    print("All models have been saved.")

    # Step 8: 保存最佳模型的预测结果
    best_model_name = results_df.iloc[0]["Model"]
    best_model = None
    if best_model_name == "Random Forest":
        best_model = rf_model
    elif best_model_name == "XGBoost":
        best_model = xgb_model
    elif best_model_name == "LightGBM":
        best_model = lgbm_model
    elif best_model_name == "Gradient Boosting":
        best_model = gb_model
    elif best_model_name == "SVM":
        best_model = svm_model
    elif best_model_name == "MLP":
        best_model = mlp_model
    elif best_model_name == "Linear Regression":
        best_model = lr_model
    elif best_model_name == "Lasso Regression":
        best_model = lasso_model

    y_pred_best = best_model.predict(X_test)
    predictions_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_best
    })
    predictions_df.to_csv("best_model_predictions.csv", index=False)
    print("Best model predictions saved to 'best_model_predictions.csv'.")

    # Step 9: 使用最佳模型对 external dataset 生成预测
    EXTERNAL_TEST_PATH = "./data/original/TestDatasetExample.xls"
    TEST_MODEL_OUTPUT_PATH = "results/test_model_outputs"
    os.makedirs(TEST_MODEL_OUTPUT_PATH, exist_ok=True)

    generate_predictions_for_external_dataset(best_model, EXTERNAL_TEST_PATH, TEST_MODEL_OUTPUT_PATH)

    # Step 10: 可视化模型比较，包括混合模型
    plt.figure(figsize=(10, 6))
    plt.bar(results_df["Model"], results_df["R²"], color='skyblue')
    plt.xlabel("Model")
    plt.ylabel("R² Score")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_performance_comparison.png")
    print("Model performance comparison saved to 'model_performance_comparison.png'.")
